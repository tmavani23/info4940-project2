import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "state.db"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                phase TEXT NOT NULL,
                current_node_id TEXT,
                title TEXT
            );

            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                parent_id TEXT,
                merge_parent_id TEXT,
                created_at TEXT NOT NULL,
                label TEXT,
                emotion_profile TEXT,
                artistic_profile TEXT,
                code TEXT,
                summary TEXT,
                confidence TEXT,
                source TEXT,
                note TEXT,
                conflict_json TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                phase TEXT NOT NULL,
                node_id TEXT,
                attachments_json TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS attachments (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_id TEXT,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                path TEXT NOT NULL,
                kind TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            """
        )


def _row_to_dict(row):
    if row is None:
        return None
    return dict(row)


def create_session(title="Untitled Session"):
    session_id = str(uuid.uuid4())
    created_at = _utc_now()
    with connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, created_at, phase, current_node_id, title) VALUES (?, ?, ?, ?, ?)",
            (session_id, created_at, "discovery", None, title),
        )
        root_node_id = create_node(
            session_id=session_id,
            parent_id=None,
            merge_parent_id=None,
            emotion_profile="",
            artistic_profile="",
            code="",
            label="root",
            summary="",
            confidence="",
            source="system",
            note="Initial root node",
            conflict=None,
            conn=conn,
        )
        conn.execute(
            "UPDATE sessions SET current_node_id = ? WHERE id = ?",
            (root_node_id, session_id),
        )
        conn.commit()
        return _row_to_dict(
            conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        )


def get_or_create_session():
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row:
            return _row_to_dict(row)

    return create_session()


def get_session(session_id):
    with connect() as conn:
        return _row_to_dict(
            conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        )


def update_session_phase(session_id, phase):
    with connect() as conn:
        conn.execute("UPDATE sessions SET phase = ? WHERE id = ?", (phase, session_id))


def update_session_current_node(session_id, node_id):
    with connect() as conn:
        conn.execute(
            "UPDATE sessions SET current_node_id = ? WHERE id = ?",
            (node_id, session_id),
        )


def create_node(
    session_id,
    parent_id,
    merge_parent_id,
    emotion_profile,
    artistic_profile,
    code,
    label,
    summary,
    confidence,
    source,
    note,
    conflict,
    conn=None,
):
    node_id = str(uuid.uuid4())
    created_at = _utc_now()
    conflict_json = json.dumps(conflict) if conflict is not None else None

    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True

    conn.execute(
        """
        INSERT INTO nodes (
            id, session_id, parent_id, merge_parent_id, created_at, label,
            emotion_profile, artistic_profile, code, summary, confidence, source, note, conflict_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            node_id,
            session_id,
            parent_id,
            merge_parent_id,
            created_at,
            label,
            emotion_profile,
            artistic_profile,
            code,
            summary,
            confidence,
            source,
            note,
            conflict_json,
        ),
    )

    if close_conn:
        conn.commit()
        conn.close()

    return node_id


def get_node(node_id):
    with connect() as conn:
        return _row_to_dict(
            conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        )


def list_nodes(session_id):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM nodes WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_messages(session_id, limit=200):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def add_message(session_id, role, content, phase, node_id=None, attachments=None):
    message_id = str(uuid.uuid4())
    created_at = _utc_now()
    attachments_json = json.dumps(attachments or [])
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO messages (id, session_id, created_at, role, content, phase, node_id, attachments_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, session_id, created_at, role, content, phase, node_id, attachments_json),
        )
    return message_id


def add_attachment(session_id, message_id, filename, content_type, path, kind):
    attachment_id = str(uuid.uuid4())
    created_at = _utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO attachments (id, session_id, message_id, created_at, filename, content_type, path, kind)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (attachment_id, session_id, message_id, created_at, filename, content_type, path, kind),
        )
    return attachment_id


def attach_to_message(attachment_id, message_id):
    with connect() as conn:
        conn.execute(
            "UPDATE attachments SET message_id = ? WHERE id = ?",
            (message_id, attachment_id),
        )


def list_attachments(session_id):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM attachments WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_attachments_by_ids(session_id, attachment_ids):
    if not attachment_ids:
        return []
    placeholders = ",".join("?" for _ in attachment_ids)
    params = [session_id] + list(attachment_ids)
    query = f"SELECT * FROM attachments WHERE session_id = ? AND id IN ({placeholders})"
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_graph(session_id):
    nodes = list_nodes(session_id)
    edges = []
    for node in nodes:
        if node.get("parent_id"):
            edges.append({"from": node["parent_id"], "to": node["id"], "type": "parent"})
        if node.get("merge_parent_id"):
            edges.append({"from": node["merge_parent_id"], "to": node["id"], "type": "merge"})
    return nodes, edges


def create_branch_from_node(
    session_id,
    target_node_id,
    note="",
    source="system",
    label=None,
    summary=None,
    confidence=None,
):
    target = get_node(target_node_id)
    if not target:
        raise ValueError("Target node not found")
    label = label or f"branch from {target_node_id[:8]}"
    summary = summary if summary is not None else (target.get("summary") or "")
    confidence = confidence if confidence is not None else (target.get("confidence") or "")
    new_id = create_node(
        session_id=session_id,
        parent_id=target_node_id,
        merge_parent_id=None,
        emotion_profile=target.get("emotion_profile") or "",
        artistic_profile=target.get("artistic_profile") or "",
        code=target.get("code") or "",
        label=label,
        summary=summary,
        confidence=confidence,
        source=source,
        note=note or "Branch created from earlier version",
        conflict=None,
    )
    update_session_current_node(session_id, new_id)
    return new_id


def create_snapshot_from_current(session_id, label, note, source, emotion_profile, artistic_profile, code, summary="", confidence="", conflict=None, merge_parent_id=None):
    session = get_session(session_id)
    parent_id = session.get("current_node_id") if session else None
    new_id = create_node(
        session_id=session_id,
        parent_id=parent_id,
        merge_parent_id=merge_parent_id,
        emotion_profile=emotion_profile,
        artistic_profile=artistic_profile,
        code=code,
        label=label,
        summary=summary,
        confidence=confidence,
        source=source,
        note=note,
        conflict=conflict,
    )
    update_session_current_node(session_id, new_id)
    return new_id


def get_current_node(session_id):
    session = get_session(session_id)
    if not session:
        return None
    node_id = session.get("current_node_id")
    if not node_id:
        return None
    return get_node(node_id)


def update_node_label_summary(node_id, label=None, summary=None):
    fields = []
    params = []
    if label is not None:
        fields.append("label = ?")
        params.append(label)
    if summary is not None:
        fields.append("summary = ?")
        params.append(summary)
    if not fields:
        return
    params.append(node_id)
    with connect() as conn:
        conn.execute(f"UPDATE nodes SET {', '.join(fields)} WHERE id = ?", params)
