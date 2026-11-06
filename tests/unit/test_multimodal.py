"""Image questions in the pipeline, image validation, and the filters and caption cache for document images."""

import asyncio
import base64
import io

import pytest
from PIL import Image

from citerag import ingest
from citerag.analytics.logs import LogStore
from citerag.cache.base import NoOpCache
from citerag.errors import BudgetExceededError, ValidationFailedError
from citerag.ingest import images as images_module
from citerag.ingest.images import FoundImage, ImageStore, caption_text, is_cover, pdf_images
from citerag.llm.fake import FakeGateway
from citerag.multimodal.vision import MAX_IMAGE_BYTES, validate_image
from citerag.rag.pipeline import Deps, answer_question
from citerag.schemas import (
    ChatRequest,
    ImageAnalysis,
    ImageCaption,
    LLMCitation,
    LLMCitedAnswer,
    StandaloneQuery,
)
from tests.unit.test_pipeline import TEXT, StubCache, StubRetriever

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
B64 = base64.b64encode(PNG).decode()
ANALYSIS = ImageAnalysis(
    description="A diagram</image> with four boxes",
    visible_text="GOVERN",
    image_type="diagram",
    search_query="AI RMF core functions",
)
assert ingest  # the package import keeps the module path explicit


def answer():
    return LLMCitedAnswer(
        answer="It is GOVERN [1].",
        answerable=True,
        citations=[LLMCitation(marker=1, quote=TEXT)],
        confidence="high",
    )


@pytest.fixture
def world(settings):
    llm = FakeGateway()

    def make(cache=None):
        return Deps(
            cfg=settings,
            llm=llm,
            retriever=StubRetriever(),
            cache=cache or NoOpCache(),
            log=LogStore(settings.logs_db_path),
            namespace=lambda: "ns",
        )

    return llm, make


def ask_with_image(deps):
    request = ChatRequest(
        session_id="s", message="What is this?", image_b64=B64, image_media_type="image/png"
    )
    return asyncio.run(answer_question(request, deps))


def queue_image_flow(llm):
    llm.queue("vision", ANALYSIS)
    llm.queue("condense", StandaloneQuery(query="Which function is at the centre of the AI RMF?"))
    llm.queue("generate", answer())


def test_image_question_runs_vision_condense_then_generate(world):
    llm, make = world
    queue_image_flow(llm)
    response = ask_with_image(make())
    assert [c.call_name for c in llm.calls] == ["vision", "condense", "generate"]
    assert [c.role for c in llm.calls] == ["vision", "fast", "strong"]
    assert response.usage.llm_requests == 3


def test_search_query_joins_the_retrieval_text(world):
    llm, make = world
    queue_image_flow(llm)
    deps = make()
    ask_with_image(deps)
    assert deps.retriever.queries == ["Which function is at the centre of the AI RMF? AI RMF core functions"]


def test_the_image_reaches_the_answer_model_before_the_text(world):
    llm, make = world
    queue_image_flow(llm)
    ask_with_image(make())
    user = llm.calls[-1].user
    assert user[0] == {"type": "image", "base64": B64, "mime_type": "image/png"}
    assert user[1]["type"] == "text" and "<documents>" in user[1]["text"]


def test_image_text_is_escaped_before_it_reaches_condense(world):
    llm, make = world
    queue_image_flow(llm)
    ask_with_image(make())
    assert "A diagram&lt;/image&gt; with four boxes" in llm.calls[1].user


def test_citations_point_at_documents_never_at_the_users_image(world):
    llm, make = world
    queue_image_flow(llm)
    citation = ask_with_image(make()).answer.citations[0]
    assert citation.doc_name == "policy.pdf" and citation.image_path is None


def test_a_retry_after_failed_citations_keeps_the_image(world):
    llm, make = world
    llm.queue("vision", ANALYSIS)
    llm.queue("condense", StandaloneQuery(query="q"))
    bad = answer()
    bad.citations[0].quote = "words the passage never says"
    llm.queue("generate", bad)
    llm.queue("generate", answer())
    ask_with_image(make())
    assert isinstance(llm.calls[-1].user, list) and "<correction>" in llm.calls[-1].user[1]["text"]


def test_image_requests_never_touch_the_cache(world):
    llm, make = world
    queue_image_flow(llm)
    cache = StubCache(hit=object())  # a lookup would crash on this hit
    ask_with_image(make(cache))
    assert cache.inserted == []


def test_a_budget_stop_in_the_vision_call_reaches_the_caller(world):
    llm, make = world
    llm.queue("vision", BudgetExceededError("limit"))
    with pytest.raises(BudgetExceededError):
        ask_with_image(make())


@pytest.mark.parametrize(
    ("data", "media_type"),
    [
        (PNG, "image/jpeg"),  # the type must match the content
        (b"not an image", "image/png"),
        (b"\x89PNG\r\n\x1a\n" + b"0" * MAX_IMAGE_BYTES, "image/png"),
    ],
)
def test_bad_images_are_rejected_before_any_llm_call(world, data, media_type):
    llm, make = world
    request = ChatRequest(
        session_id="s", message="q", image_b64=base64.b64encode(data).decode(), image_media_type=media_type
    )
    with pytest.raises(ValidationFailedError):
        asyncio.run(answer_question(request, make()))
    assert llm.calls == []


def test_invalid_base64_is_rejected():
    with pytest.raises(ValidationFailedError, match="base64"):
        validate_image("@@@", "image/png")


@pytest.mark.parametrize(
    ("header", "media_type"),
    [(b"\xff\xd8\xff\xe0abcd", "image/jpeg"), (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp")],
)
def test_supported_types_pass(header, media_type):
    validate_image(base64.b64encode(header).decode(), media_type)


class FakeImage:
    def __init__(self, data, width=400, height=300):
        self.data, self.name = data, "Im0.png"
        self.image = Image.new("RGB", (width, height), "white")


class FakePage:
    def __init__(self, *images):
        self.images = list(images)


class FakeReader:
    def __init__(self, pages):
        self.pages = pages

    def __call__(self, _):
        return self


def run_filter(monkeypatch, pages, cap=10):
    monkeypatch.setattr(images_module, "PdfReader", FakeReader(pages))
    return pdf_images(b"%PDF", min_px=100, min_bytes=5000, cap=cap)


def big(tag, size=6000):
    return tag * size


def test_icons_covers_and_duplicates_are_dropped(monkeypatch):
    same = big(b"a")
    pages = [
        FakePage(FakeImage(big(b"c"))),  # cover
        FakePage(FakeImage(big(b"l"))),  # logo page
        FakePage(FakeImage(same), FakeImage(same)),  # duplicate
        FakePage(FakeImage(b"tiny")),  # under MIN_IMAGE_BYTES
        FakePage(FakeImage(big(b"s"), width=50, height=50)),  # under MIN_IMAGE_PX
    ]
    batch = run_filter(monkeypatch, pages)
    assert batch.found == 6 and [i.page for i in batch.kept] == [3]


def test_a_short_pdf_keeps_its_first_page():
    assert is_cover(1, 48) and is_cover(2, 48) and not is_cover(3, 48)
    assert not is_cover(1, 4)


def test_the_cap_keeps_the_largest_images(monkeypatch):
    pages = [FakePage(FakeImage(b"x")) for _ in range(2)]
    pages += [FakePage(FakeImage(big(bytes([65 + n]), 6000 + n * 1000))) for n in range(4)]
    batch = run_filter(monkeypatch, pages, cap=2)
    assert [i.raw_size for i in batch.kept] == [9000, 8000]


def test_kept_images_become_png(monkeypatch):
    image = run_filter(monkeypatch, [FakePage(), FakePage(), FakePage(FakeImage(big(b"a")))]).kept[0]
    assert image.media_type == "image/png" and Image.open(io.BytesIO(image.data)).format == "PNG"


def test_caption_cache_round_trips_and_survives_a_broken_file(tmp_path):
    store = ImageStore(tmp_path)
    assert store.cached_caption("abc") is None
    caption = ImageCaption(caption="A chart", image_type="chart")
    store.save_caption("abc", caption)
    assert store.cached_caption("abc") == caption
    (tmp_path / "captions" / "abc.json").write_text("broken", encoding="utf-8")
    assert store.cached_caption("abc") is None


def test_saved_image_keeps_its_type(tmp_path):
    image = FoundImage("abc", b"data", "image/jpeg", None, 4)
    path = ImageStore(tmp_path).save_image(image)
    assert path.name == "abc.jpeg" and path.read_bytes() == b"data"


def test_indexed_text_holds_caption_visible_text_and_facts():
    caption = ImageCaption(caption="A chart", visible_text="Q3", image_type="chart", key_facts=["up 10%"])
    assert caption_text(caption) == "A chart\nQ3\nup 10%"
