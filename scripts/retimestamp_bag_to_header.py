#!/usr/bin/env python3

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from rosbags.typesys import Stores, get_typestore


def header_stamp_ns(msg: object) -> int | None:
    header = getattr(msg, "header", None)
    if header is None:
        return None
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    return int(stamp.sec) * 10**9 + int(stamp.nanosec)


def retimestamp_bag(db3_path: Path) -> None:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    conn = sqlite3.connect(str(db3_path))
    conn.row_factory = sqlite3.Row

    try:
        topics = {
            row["id"]: row["type"]
            for row in conn.execute("SELECT id, type FROM topics")
        }

        cursor = conn.execute("SELECT id, topic_id, data FROM messages ORDER BY id")
        updates: list[tuple[int, int]] = []
        changed = 0

        for row in cursor:
            msgtype = topics[row["topic_id"]]
            msg = typestore.deserialize_cdr(row["data"], msgtype)
            stamp_ns = header_stamp_ns(msg)
            if stamp_ns is None:
                continue
            updates.append((stamp_ns, row["id"]))
            changed += 1
            if len(updates) >= 1000:
                conn.executemany("UPDATE messages SET timestamp = ? WHERE id = ?", updates)
                conn.commit()
                updates.clear()

        if updates:
            conn.executemany("UPDATE messages SET timestamp = ? WHERE id = ?", updates)
            conn.commit()

        print(f"retimestamped {changed} messages in {db3_path}")
    finally:
        conn.close()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: retimestamp_bag_to_header.py <bag_db3_path>", file=sys.stderr)
        return 1

    db3_path = Path(sys.argv[1])
    if not db3_path.is_file():
        print(f"missing db3 file: {db3_path}", file=sys.stderr)
        return 1

    retimestamp_bag(db3_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
