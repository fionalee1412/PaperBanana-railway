"""
FastAPI layer for PaperVizAgent — one endpoint per agent + full pipeline.
"""

import asyncio
import json
import os
import traceback
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.retriever_agent import RetrieverAgent
from agents.planner_agent import PlannerAgent
from agents.stylist_agent import StylistAgent
from agents.visualizer_agent import VisualizerAgent
from agents.critic_agent import CriticAgent
from agents.vanilla_agent import VanillaAgent
from agents.polish_agent import PolishAgent
from utils.config import ExpConfig
from utils.paperviz_processor import PaperVizProcessor

app = FastAPI(title="PaperVizAgent API")

WORK_DIR = Path(__file__).parent
API_KEY = os.environ.get("API_KEY")


async def verify_api_key(request: Request):
    """Verify Bearer token if API_KEY is configured."""
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_config(
    task_name: str = "diagram",
    retrieval_setting: str = "auto",
    exp_mode: str = "dev_full",
    max_critic_rounds: int = 3,
    model_name: str = "",
) -> ExpConfig:
    return ExpConfig(
        dataset_name="PaperBananaBench",
        task_name=task_name,
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        max_critic_rounds=max_critic_rounds,
        model_name=model_name,
        work_dir=WORK_DIR,
    )


def _sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Request / Response Models ────────────────────────────────────────────────

class RetrieverRequest(BaseModel):
    content: Any
    visual_intent: str
    task_name: Literal["diagram", "plot"] = "diagram"
    retrieval_setting: Literal["auto", "manual", "random", "none"] = "auto"

class RetrieverResponse(BaseModel):
    top10_references: List[str]
    retrieved_examples: List[Any]


class PlannerRequest(BaseModel):
    content: Any
    visual_intent: str
    task_name: Literal["diagram", "plot"] = "diagram"
    top10_references: Optional[List[str]] = []
    retrieved_examples: Optional[List[Any]] = []
    retrieval_setting: Literal["auto", "manual", "random", "none"] = "auto"

class PlannerResponse(BaseModel):
    description: str


class StylistRequest(BaseModel):
    content: Any
    visual_intent: str
    task_name: Literal["diagram", "plot"] = "diagram"
    description: str

class StylistResponse(BaseModel):
    styled_description: str


class VisualizerRequest(BaseModel):
    description: str
    task_name: Literal["diagram", "plot"] = "diagram"
    aspect_ratio: Optional[str] = "1:1"
    model_name: Optional[str] = ""

class VisualizerResponse(BaseModel):
    image_base64: Optional[str] = None


class CriticRequest(BaseModel):
    content: Any
    visual_intent: str
    task_name: Literal["diagram", "plot"] = "diagram"
    description: str
    image_base64: Optional[str] = None

class CriticResponse(BaseModel):
    critic_suggestions: str
    revised_description: str


class VanillaRequest(BaseModel):
    content: Any
    visual_intent: str
    task_name: Literal["diagram", "plot"] = "diagram"
    aspect_ratio: Optional[str] = "1:1"

class VanillaResponse(BaseModel):
    image_base64: Optional[str] = None


class PolishRequest(BaseModel):
    image_base64: str
    task_name: Literal["diagram", "plot"] = "diagram"
    aspect_ratio: Optional[str] = "16:9"

class PolishResponse(BaseModel):
    suggestions: str
    polished_image_base64: Optional[str] = None


class PipelineRequest(BaseModel):
    content: Any
    visual_intent: str
    task_name: Literal["diagram", "plot"] = "diagram"
    exp_mode: str = "demo_full"
    retrieval_setting: Literal["auto", "manual", "random", "none"] = "auto"
    aspect_ratio: Optional[str] = "1:1"
    max_critic_rounds: Optional[int] = 3


class PipelineResponse(BaseModel):
    image_base64: Optional[str] = None
    descriptions: Dict[str, str] = {}
    stages: Dict[str, Any] = {}


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/agents/retriever", response_model=RetrieverResponse, dependencies=[Depends(verify_api_key)])
async def run_retriever(req: RetrieverRequest):
    try:
        cfg = _make_config(task_name=req.task_name, retrieval_setting=req.retrieval_setting)
        agent = RetrieverAgent(exp_config=cfg)
        data: Dict[str, Any] = {"content": req.content, "visual_intent": req.visual_intent}
        result = await agent.process(data, retrieval_setting=req.retrieval_setting)
        return RetrieverResponse(
            top10_references=result.get("top10_references", []),
            retrieved_examples=result.get("retrieved_examples", []),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/planner", response_model=PlannerResponse, dependencies=[Depends(verify_api_key)])
async def run_planner(req: PlannerRequest):
    try:
        cfg = _make_config(task_name=req.task_name, retrieval_setting=req.retrieval_setting)
        agent = PlannerAgent(exp_config=cfg)
        data: Dict[str, Any] = {
            "content": req.content,
            "visual_intent": req.visual_intent,
            "top10_references": req.top10_references or [],
            "retrieved_examples": req.retrieved_examples or [],
        }
        result = await agent.process(data)
        desc_key = f"target_{req.task_name}_desc0"
        return PlannerResponse(description=result.get(desc_key, ""))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/stylist", response_model=StylistResponse, dependencies=[Depends(verify_api_key)])
async def run_stylist(req: StylistRequest):
    try:
        cfg = _make_config(task_name=req.task_name)
        agent = StylistAgent(exp_config=cfg)
        desc_key = f"target_{req.task_name}_desc0"
        data: Dict[str, Any] = {
            "content": req.content,
            "visual_intent": req.visual_intent,
            desc_key: req.description,
        }
        result = await agent.process(data)
        styled_key = f"target_{req.task_name}_stylist_desc0"
        return StylistResponse(styled_description=result.get(styled_key, ""))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/visualizer", response_model=VisualizerResponse, dependencies=[Depends(verify_api_key)])
async def run_visualizer(req: VisualizerRequest):
    try:
        cfg = _make_config(task_name=req.task_name, model_name=req.model_name or "")
        agent = VisualizerAgent(exp_config=cfg)
        desc_key = f"target_{req.task_name}_desc0"
        data: Dict[str, Any] = {
            desc_key: req.description,
            "additional_info": {"rounded_ratio": req.aspect_ratio or "1:1"},
        }
        result = await agent.process(data)
        image_key = f"{desc_key}_base64_jpg"
        return VisualizerResponse(image_base64=result.get(image_key))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/critic", response_model=CriticResponse, dependencies=[Depends(verify_api_key)])
async def run_critic(req: CriticRequest):
    try:
        cfg = _make_config(task_name=req.task_name)
        agent = CriticAgent(exp_config=cfg)
        desc_key = f"target_{req.task_name}_desc0"
        base64_key = f"{desc_key}_base64_jpg"
        data: Dict[str, Any] = {
            "content": req.content,
            "visual_intent": req.visual_intent,
            desc_key: req.description,
            base64_key: req.image_base64 or "",
            "current_critic_round": 0,
        }
        result = await agent.process(data, source="planner")
        task = req.task_name
        return CriticResponse(
            critic_suggestions=result.get(f"target_{task}_critic_suggestions0", ""),
            revised_description=result.get(f"target_{task}_critic_desc0", ""),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/vanilla", response_model=VanillaResponse, dependencies=[Depends(verify_api_key)])
async def run_vanilla(req: VanillaRequest):
    try:
        cfg = _make_config(task_name=req.task_name)
        agent = VanillaAgent(exp_config=cfg)
        data: Dict[str, Any] = {
            "content": req.content,
            "visual_intent": req.visual_intent,
            "additional_info": {"rounded_ratio": req.aspect_ratio or "1:1"},
        }
        result = await agent.process(data)
        image_key = f"vanilla_{req.task_name}_base64_jpg"
        return VanillaResponse(image_base64=result.get(image_key))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/polish", response_model=PolishResponse, dependencies=[Depends(verify_api_key)])
async def run_polish(req: PolishRequest):
    try:
        cfg = _make_config(task_name=req.task_name)
        agent = PolishAgent(exp_config=cfg)
        task = req.task_name
        data: Dict[str, Any] = {
            "path_to_gt_image": None,
            "additional_info": {"rounded_ratio": req.aspect_ratio or "16:9"},
        }
        style_guide_path = WORK_DIR / "style_guides" / agent.style_guide_filename
        with open(style_guide_path, "r", encoding="utf-8") as f:
            style_guide = f.read()

        suggestions = await agent._generate_suggestions(req.image_base64, style_guide)
        data[f"suggestions_{task}"] = suggestions

        from utils import generation_utils, image_utils
        content_list = [
            {"type": "text", "text": f"Please polish this image based on the following suggestions:\n\n{suggestions}\n\nPolished Image:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": req.image_base64}},
        ]
        response_list = await generation_utils.call_gemini_with_retry_async(
            model_name=agent.image_model_name,
            contents=content_list,
            config={
                "system_instruction": agent.system_prompt,
                "temperature": cfg.temperature,
                "candidate_count": 1,
                "max_output_tokens": 50000,
                "response_modalities": ["IMAGE"],
                "image_config": {
                    "aspect_ratio": req.aspect_ratio or "16:9",
                    "image_size": "1K",
                },
            },
            max_attempts=5,
            retry_delay=30,
        )
        polished_b64 = None
        if response_list and response_list[0]:
            polished_b64 = image_utils.convert_png_b64_to_jpg_b64(response_list[0])

        return PolishResponse(
            suggestions=suggestions or "",
            polished_image_base64=polished_b64,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Pipeline (SSE streaming to avoid CDN timeout) ───────────────────────────

@app.post("/pipeline", dependencies=[Depends(verify_api_key)])
async def run_pipeline(req: PipelineRequest):
    """
    Run the full pipeline as a Server-Sent Events stream.
    Sends progress events as each agent completes, preventing CDN timeout.

    Events:
      - event: stage   → {"stage": "retriever", "status": "done", ...}
      - event: stage   → {"stage": "planner", "status": "done", ...}
      - ...
      - event: result  → full PipelineResponse JSON
      - event: error   → {"error": "..."}
    """

    async def _stream() -> AsyncGenerator[str, None]:
        try:
            cfg = _make_config(
                task_name=req.task_name,
                retrieval_setting=req.retrieval_setting,
                exp_mode=req.exp_mode,
                max_critic_rounds=req.max_critic_rounds or 3,
            )
            task = req.task_name
            retrieval_setting = req.retrieval_setting
            max_rounds = req.max_critic_rounds or 3

            data: Dict[str, Any] = {
                "content": req.content,
                "visual_intent": req.visual_intent,
                "additional_info": {"rounded_ratio": req.aspect_ratio or "1:1"},
                "max_critic_rounds": max_rounds,
            }

            exp_mode = req.exp_mode

            # ── Vanilla mode ──
            if exp_mode == "vanilla":
                agent = VanillaAgent(exp_config=cfg)
                data = await agent.process(data)
                data["eval_image_field"] = f"vanilla_{task}_base64_jpg"
                yield _sse_event("stage", {"stage": "vanilla", "status": "done"})

            # ── Planner only ──
            elif exp_mode == "dev_planner":
                retriever = RetrieverAgent(exp_config=cfg)
                data = await retriever.process(data, retrieval_setting=retrieval_setting)
                yield _sse_event("stage", {"stage": "retriever", "status": "done"})

                planner = PlannerAgent(exp_config=cfg)
                data = await planner.process(data)
                yield _sse_event("stage", {"stage": "planner", "status": "done"})

                visualizer = VisualizerAgent(exp_config=cfg)
                data = await visualizer.process(data)
                data["eval_image_field"] = f"target_{task}_desc0_base64_jpg"
                yield _sse_event("stage", {"stage": "visualizer", "status": "done"})

            # ── Planner + Stylist ──
            elif exp_mode == "dev_planner_stylist":
                retriever = RetrieverAgent(exp_config=cfg)
                data = await retriever.process(data, retrieval_setting=retrieval_setting)
                yield _sse_event("stage", {"stage": "retriever", "status": "done"})

                planner = PlannerAgent(exp_config=cfg)
                data = await planner.process(data)
                yield _sse_event("stage", {"stage": "planner", "status": "done"})

                stylist = StylistAgent(exp_config=cfg)
                data = await stylist.process(data)
                yield _sse_event("stage", {"stage": "stylist", "status": "done"})

                visualizer = VisualizerAgent(exp_config=cfg)
                data = await visualizer.process(data)
                data["eval_image_field"] = f"target_{task}_stylist_desc0_base64_jpg"
                yield _sse_event("stage", {"stage": "visualizer", "status": "done"})

            # ── Planner + Critic ──
            elif exp_mode in ["dev_planner_critic", "demo_planner_critic"]:
                retriever = RetrieverAgent(exp_config=cfg)
                data = await retriever.process(data, retrieval_setting=retrieval_setting)
                yield _sse_event("stage", {"stage": "retriever", "status": "done"})

                planner = PlannerAgent(exp_config=cfg)
                data = await planner.process(data)
                yield _sse_event("stage", {"stage": "planner", "status": "done"})

                visualizer = VisualizerAgent(exp_config=cfg)
                data = await visualizer.process(data)
                yield _sse_event("stage", {"stage": "visualizer", "status": "done"})

                critic = CriticAgent(exp_config=cfg)
                current_best_key = f"target_{task}_desc0_base64_jpg"
                for round_idx in range(max_rounds):
                    data["current_critic_round"] = round_idx
                    data = await critic.process(data, source="planner")
                    sug = data.get(f"target_{task}_critic_suggestions{round_idx}", "")
                    if sug.strip() == "No changes needed.":
                        yield _sse_event("stage", {"stage": f"critic_round_{round_idx}", "status": "no_changes"})
                        break
                    data = await visualizer.process(data)
                    new_key = f"target_{task}_critic_desc{round_idx}_base64_jpg"
                    if new_key in data and data[new_key]:
                        current_best_key = new_key
                    yield _sse_event("stage", {"stage": f"critic_round_{round_idx}", "status": "done"})
                data["eval_image_field"] = current_best_key

            # ── Full pipeline ──
            elif exp_mode in ["dev_full", "demo_full"]:
                retriever = RetrieverAgent(exp_config=cfg)
                data = await retriever.process(data, retrieval_setting=retrieval_setting)
                yield _sse_event("stage", {"stage": "retriever", "status": "done"})

                planner = PlannerAgent(exp_config=cfg)
                data = await planner.process(data)
                yield _sse_event("stage", {"stage": "planner", "status": "done"})

                stylist = StylistAgent(exp_config=cfg)
                data = await stylist.process(data)
                yield _sse_event("stage", {"stage": "stylist", "status": "done"})

                visualizer = VisualizerAgent(exp_config=cfg)
                data = await visualizer.process(data)
                yield _sse_event("stage", {"stage": "visualizer", "status": "done"})

                critic = CriticAgent(exp_config=cfg)
                current_best_key = f"target_{task}_stylist_desc0_base64_jpg"
                for round_idx in range(max_rounds):
                    data["current_critic_round"] = round_idx
                    data = await critic.process(data, source="stylist")
                    sug = data.get(f"target_{task}_critic_suggestions{round_idx}", "")
                    if sug.strip() == "No changes needed.":
                        yield _sse_event("stage", {"stage": f"critic_round_{round_idx}", "status": "no_changes"})
                        break
                    data = await visualizer.process(data)
                    new_key = f"target_{task}_critic_desc{round_idx}_base64_jpg"
                    if new_key in data and data[new_key]:
                        current_best_key = new_key
                    yield _sse_event("stage", {"stage": f"critic_round_{round_idx}", "status": "done"})
                data["eval_image_field"] = current_best_key

            # ── Polish ──
            elif exp_mode == "dev_polish":
                agent = PolishAgent(exp_config=cfg)
                data = await agent.process(data)
                data["eval_image_field"] = f"polished_{task}_base64_jpg"
                yield _sse_event("stage", {"stage": "polish", "status": "done"})

            else:
                yield _sse_event("error", {"error": f"Unknown exp_mode: {exp_mode}"})
                return

            # ── Build final response ──
            eval_field = data.get("eval_image_field", "")
            image_b64 = data.get(eval_field) if eval_field else None

            descriptions: Dict[str, str] = {}
            for key in [
                f"target_{task}_desc0",
                f"target_{task}_stylist_desc0",
                f"target_{task}_critic_desc0",
                f"target_{task}_critic_desc1",
                f"target_{task}_critic_desc2",
            ]:
                if key in data:
                    descriptions[key] = data[key]

            stages: Dict[str, Any] = {}
            if "top10_references" in data:
                stages["retriever"] = {"top10_references": data["top10_references"]}
            for key in list(data.keys()):
                if "critic_suggestions" in key:
                    stages[key] = data[key]

            result = PipelineResponse(
                image_base64=image_b64,
                descriptions=descriptions,
                stages=stages,
            )
            yield _sse_event("result", result.model_dump())

        except Exception as e:
            traceback.print_exc()
            yield _sse_event("error", {"error": str(e)})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
