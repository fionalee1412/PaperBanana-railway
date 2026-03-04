"""
FastAPI layer for PaperVizAgent — one endpoint per agent + full pipeline.
"""

import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
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
            "path_to_gt_image": None,  # not used when image_base64 provided directly
            "additional_info": {"rounded_ratio": req.aspect_ratio or "16:9"},
        }
        # Polish agent normally loads from file — override _generate_suggestions
        # to work with the provided base64 directly
        style_guide_path = WORK_DIR / "style_guides" / agent.style_guide_filename
        with open(style_guide_path, "r", encoding="utf-8") as f:
            style_guide = f.read()

        suggestions = await agent._generate_suggestions(req.image_base64, style_guide)
        data[f"suggestions_{task}"] = suggestions

        # Step 2: generate polished image
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


@app.post("/pipeline", response_model=PipelineResponse, dependencies=[Depends(verify_api_key)])
async def run_pipeline(req: PipelineRequest):
    try:
        cfg = _make_config(
            task_name=req.task_name,
            retrieval_setting=req.retrieval_setting,
            exp_mode=req.exp_mode,
            max_critic_rounds=req.max_critic_rounds or 3,
        )
        processor = PaperVizProcessor(
            exp_config=cfg,
            vanilla_agent=VanillaAgent(exp_config=cfg),
            planner_agent=PlannerAgent(exp_config=cfg),
            visualizer_agent=VisualizerAgent(exp_config=cfg),
            stylist_agent=StylistAgent(exp_config=cfg),
            critic_agent=CriticAgent(exp_config=cfg),
            retriever_agent=RetrieverAgent(exp_config=cfg),
            polish_agent=PolishAgent(exp_config=cfg),
        )
        data: Dict[str, Any] = {
            "content": req.content,
            "visual_intent": req.visual_intent,
            "additional_info": {"rounded_ratio": req.aspect_ratio or "1:1"},
            "max_critic_rounds": req.max_critic_rounds or 3,
        }
        result = await processor.process_single_query(data, do_eval=False)

        # Extract final image from eval_image_field
        eval_field = result.get("eval_image_field", "")
        image_b64 = result.get(eval_field) if eval_field else None

        # Collect descriptions
        task = req.task_name
        descriptions: Dict[str, str] = {}
        for key in [
            f"target_{task}_desc0",
            f"target_{task}_stylist_desc0",
            f"target_{task}_critic_desc0",
            f"target_{task}_critic_desc1",
            f"target_{task}_critic_desc2",
        ]:
            if key in result:
                descriptions[key] = result[key]

        # Collect stage metadata
        stages: Dict[str, Any] = {}
        if "top10_references" in result:
            stages["retriever"] = {"top10_references": result["top10_references"]}
        for key in list(result.keys()):
            if "critic_suggestions" in key:
                stages[key] = result[key]

        return PipelineResponse(
            image_base64=image_b64,
            descriptions=descriptions,
            stages=stages,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
