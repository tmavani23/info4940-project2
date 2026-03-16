import json
import mimetypes
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from .db import (
    add_attachment,
    add_message,
    attach_to_message,
    create_session,
    create_branch_from_node,
    create_snapshot_from_current,
    get_node,
    get_attachments_by_ids,
    get_current_node,
    get_graph,
    get_or_create_session,
    get_session,
    init_db,
    list_attachments,
    list_messages,
    list_nodes,
    update_node_label_summary,
    update_session_current_node,
    update_session_phase,
)
from .llm import (
    generate_reply,
    repair_json_response,
    generate_transition_message,
    generate_commit_summary,
)
from .utils import (
    detect_conflict,
    extract_confidence,
    extract_intro_emotion,
    extract_p5js_code,
    extract_section,
    is_likely_complete_code,
    find_last_valid_code,
    parse_llm_json,
)

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.on_event("startup")
def _startup():
    init_db()
    get_or_create_session()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/debug", response_class=HTMLResponse)
def debug(request: Request):
    return templates.TemplateResponse("debug.html", {"request": request})


@app.get("/api/state")
def api_state():
    session = get_or_create_session()
    current_node = get_current_node(session["id"])
    nodes = list_nodes(session["id"])
    nodes = _backfill_commit_summaries(nodes)
    messages = list_messages(session["id"])
    attachments = list_attachments(session["id"])
    return {
        "session": session,
        "current_node": current_node,
        "nodes": nodes,
        "messages": messages,
        "attachments": attachments,
    }


@app.get("/api/graph")
def api_graph():
    session = get_or_create_session()
    nodes, edges = get_graph(session["id"])
    return {"nodes": nodes, "edges": edges}


@app.post("/api/phase")
def api_phase(payload: dict):
    session = get_or_create_session()
    phase = payload.get("phase", "discovery")
    if phase not in ("discovery", "implementation"):
        return JSONResponse({"error": "Invalid phase"}, status_code=400)
    update_session_phase(session["id"], phase)
    return {"ok": True, "phase": phase}


@app.post("/api/attachments")
async def api_attachments(file: UploadFile = File(...)):
    session = get_or_create_session()
    contents = await file.read()
    if not contents:
        return JSONResponse({"error": "Empty file"}, status_code=400)

    safe_name = file.filename.replace("/", "_")
    attachment_id = add_attachment(
        session_id=session["id"],
        message_id=None,
        filename=safe_name,
        content_type=file.content_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream",
        path="",
        kind=_infer_kind(file.content_type or ""),
    )
    path = UPLOAD_DIR / f"{attachment_id}_{safe_name}"
    path.write_bytes(contents)

    # Update path in DB
    update_attachment_path(attachment_id, str(path))

    return {
        "id": attachment_id,
        "filename": safe_name,
        "content_type": file.content_type,
        "path": f"/uploads/{path.name}",
    }


@app.post("/api/chat")
def api_chat(payload: dict):
    session = get_or_create_session()
    phase = payload.get("phase") or session.get("phase") or "discovery"
    learning_mode = bool(payload.get("learning_mode"))
    message = (payload.get("message") or "").strip()
    attachment_ids = payload.get("attachment_ids") or []

    if not message:
        return JSONResponse({"error": "Message required"}, status_code=400)

    # Add user message
    message_id = add_message(
        session["id"],
        "user",
        message,
        phase,
        node_id=session.get("current_node_id"),
        attachments=attachment_ids,
    )
    if attachment_ids:
        for att_id in attachment_ids:
            attach_to_message(att_id, message_id)

    # Build conversation context (last 12 messages + current user message)
    history = list_messages(session["id"], limit=50)
    recent = history[-12:] if len(history) > 12 else history
    convo = [{"role": m["role"], "content": m["content"]} for m in recent]

    # Fetch attachments for LLM
    attachments = get_attachments_by_ids(session["id"], attachment_ids)

    current_node = get_current_node(session["id"])
    reply = generate_reply(phase, convo, current_node or {}, learning_mode=learning_mode, attachments=attachments)

    parsed = parse_llm_json(reply)
    if not parsed:
        repair_text = repair_json_response(reply, phase)
        parsed = parse_llm_json(repair_text)

    if parsed:
        def _safe_str(value):
            if value is None:
                return ""
            if isinstance(value, (list, dict)):
                return ""
            return str(value)

        assistant_message = _safe_str(parsed.get("assistant_message")).strip()
        questions = parsed.get("questions") or []
        if isinstance(questions, list) and questions:
            questions_str = [str(q) for q in questions if q is not None]
            assistant_message = assistant_message + "\n" + "\n".join(questions_str)
        emotion_profile = _safe_str(parsed.get("emotion_profile")).strip()
        artistic_profile = _safe_str(parsed.get("artistic_profile")).strip()
        code_block = _safe_str(parsed.get("p5js_code")).strip()
        confidence = _safe_str(parsed.get("confidence")).strip()
        reply_for_user = assistant_message or "Updated the state."
    else:
        reply_for_user = (
            "System: The model did not return valid JSON. Please retry your request."
        )
        emotion_profile = ""
        artistic_profile = ""
        code_block = ""
        confidence = ""

    if code_block and not is_likely_complete_code(code_block):
        code_block = ""

    # Store assistant message (chat-friendly only)
    add_message(session["id"], "assistant", reply_for_user, phase, node_id=session.get("current_node_id"))

    if emotion_profile or artistic_profile or code_block:
        # Build new snapshot using existing state as base
        base = current_node or {}
        new_emotion = emotion_profile or base.get("emotion_profile") or ""
        new_artistic = artistic_profile or base.get("artistic_profile") or ""

        nodes = list_nodes(session["id"])
        last_valid_code = find_last_valid_code(nodes)
        base_code = base.get("code") or ""
        if not is_likely_complete_code(base_code) and last_valid_code:
            base_code = last_valid_code

        code_valid = True if code_block else False
        if code_block and not is_likely_complete_code(code_block):
            code_valid = False
            code_block = ""

        new_code = code_block or base_code or ""

        note = "Auto snapshot from assistant response"
        conflict = None
        if detect_conflict(reply):
            conflict = {
                "detected": True,
                "note": "Assistant flagged a possible mismatch. Review and branch if needed.",
            }
        if code_block == "" and not code_valid:
            conflict = {
                "detected": True,
                "note": "Code update skipped because the response looked incomplete or truncated.",
            }

        after_state = {
            "emotion_profile": new_emotion,
            "artistic_profile": new_artistic,
            "code": new_code,
        }
        commit_summary = generate_commit_summary("assistant-update", base, after_state)
        label = commit_summary or "assistant update"
        create_snapshot_from_current(
            session_id=session["id"],
            label=label,
            note=note,
            source="assistant",
            emotion_profile=new_emotion,
            artistic_profile=new_artistic,
            code=new_code,
            summary=commit_summary or "",
            confidence=confidence,
            conflict=conflict,
        )

    return {"reply": reply}


@app.post("/api/node")
def api_node(payload: dict):
    session = get_or_create_session()
    base = get_current_node(session["id"]) or {}
    new_emotion = payload.get("emotion_profile", base.get("emotion_profile") or "")
    new_artistic = payload.get("artistic_profile", base.get("artistic_profile") or "")
    new_code = payload.get("code", base.get("code") or "")
    after_state = {
        "emotion_profile": new_emotion,
        "artistic_profile": new_artistic,
        "code": new_code,
    }
    commit_summary = generate_commit_summary("user-update", base, after_state)
    label = commit_summary or payload.get("label", "manual snapshot")
    note = payload.get("note", "")
    source = payload.get("source", "user")

    node_id = create_snapshot_from_current(
        session_id=session["id"],
        label=label,
        note=note,
        source=source,
        emotion_profile=new_emotion,
        artistic_profile=new_artistic,
        code=new_code,
        summary=commit_summary or "",
    )
    return {"ok": True, "node_id": node_id}


@app.post("/api/revert")
def api_revert(payload: dict):
    session = get_or_create_session()
    target_id = payload.get("target_node_id")
    if not target_id:
        return JSONResponse({"error": "target_node_id required"}, status_code=400)
    note = payload.get("note", "Reverted and branched")
    try:
        current_node = get_current_node(session["id"]) or {}
        target_node = get_node(target_id) or {}
        commit_summary = generate_commit_summary("revert", current_node, target_node)
        new_id = create_branch_from_node(
            session["id"],
            target_id,
            note=note,
            source="user",
            label=commit_summary or None,
            summary=commit_summary or "",
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    message = generate_transition_message("revert", target_node)
    add_message(
        session["id"],
        "assistant",
        message,
        session.get("phase") or "discovery",
        node_id=new_id,
    )
    return {"ok": True, "node_id": new_id}


@app.post("/api/set-current")
def api_set_current(payload: dict):
    session = get_or_create_session()
    node_id = payload.get("node_id")
    if not node_id:
        return JSONResponse({"error": "node_id required"}, status_code=400)
    update_session_current_node(session["id"], node_id)
    target_node = get_node(node_id) or {}
    message = generate_transition_message("set-current", target_node)
    add_message(
        session["id"],
        "assistant",
        message,
        session.get("phase") or "discovery",
        node_id=node_id,
    )
    return {"ok": True}


@app.post("/api/reset")
def api_reset():
    session = create_session()
    return {"ok": True, "session_id": session["id"]}


@app.post("/api/new-session")
def api_new_session():
    session = create_session()
    return {"ok": True, "session_id": session["id"]}


# --- helpers ---


def _infer_kind(content_type):
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("audio/"):
        return "audio"
    return "file"


def update_attachment_path(attachment_id, path_str):
    from .db import connect

    with connect() as conn:
        conn.execute(
            "UPDATE attachments SET path = ? WHERE id = ?",
            (path_str, attachment_id),
        )


def _is_generic_label(label):
    clean = (label or "").strip().lower()
    if not clean:
        return True
    if clean.startswith("branch from"):
        return True
    return clean in {"assistant update", "snapshot", "manual snapshot", "profile update", "root"}


def _infer_backfill_action(node):
    note = (node.get("note") or "").lower()
    label = (node.get("label") or "").lower()
    if "revert" in note or "reverted" in note or label.startswith("branch from"):
        return "revert"
    return "backfill"


def _backfill_commit_summaries(nodes):
    if not nodes:
        return nodes
    node_map = {n["id"]: n for n in nodes}
    remaining = 6
    for node in nodes:
        if remaining <= 0:
            break
        summary = (node.get("summary") or "").strip()
        label = (node.get("label") or "").strip()
        if summary and label and not _is_generic_label(label):
            continue
        parent_id = node.get("parent_id")
        if not parent_id:
            continue
        parent = node_map.get(parent_id)
        if not parent:
            continue
        action = _infer_backfill_action(node)
        commit_summary = generate_commit_summary(action, parent, node)
        if not commit_summary:
            continue
        update_node_label_summary(node["id"], label=commit_summary, summary=commit_summary)
        node["label"] = commit_summary
        node["summary"] = commit_summary
        remaining -= 1
    return nodes
