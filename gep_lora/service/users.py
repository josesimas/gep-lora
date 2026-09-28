"""
users.py - The people allowed to submit jobs, and their API keys.

    python -m gep_lora.service.users add alice       # prints alice's key, once
    python -m gep_lora.service.users rotate alice    # a new key; the old one stops working
    python -m gep_lora.service.users remove alice    # the registry forgets their jobs; files stay
    python -m gep_lora.service.users list

Only a hash of a key is stored, so a lost key is rotated, never recovered.
"""

import argparse
import sqlite3
import sys

from gep_lora.service import settings
from gep_lora.service.registry import Registry


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manage async API users.")
    parser.add_argument("--jobs-dir", default=None,
                        help="the registry folder (default %s)" % settings.JOBS_DIR)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("add", "rotate", "remove"):
        sub.add_parser(action).add_argument("name")
    sub.add_parser("list")
    args = parser.parse_args(argv)
    registry = Registry(args.jobs_dir)

    if args.action == "list":
        for user in registry.users():
            print("%-4d %-20s %s" % (user["id"], user["name"], user["created_at"]))
        return 0
    if args.action == "add":
        try:
            key = registry.add_user(args.name)
        except sqlite3.IntegrityError:
            print("there is already a user called %s" % args.name)
            return 1
        print(key)
        return 0
    if args.action == "rotate":
        key = registry.rotate_key(args.name)
        if key is None:
            print("no user called %s" % args.name)
            return 1
        print(key)
        return 0
    # remove: the jobs' files stay on disk; only the registry forgets them.
    if not registry.remove_user(args.name):
        print("no user called %s" % args.name)
        return 1
    print("removed %s" % args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
