#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-bin", type=Path, required=True)
    ap.add_argument("--output-bin", type=Path, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--expected-output", type=int, default=0)
    ap.add_argument("--chunk-records", type=int, default=4096)
    args = ap.parse_args()

    if args.stride <= 0:
        raise ValueError("stride must be positive")
    if args.offset < 0 or args.offset >= args.stride:
        raise ValueError("offset must be in [0, stride)")

    with args.input_bin.open("rb") as src:
        header = src.read(8)
        if len(header) != 8:
            raise ValueError(f"bad input header: {args.input_bin}")
        n, dim = struct.unpack("<II", header)
        record_bytes = dim * 4
        out_n = 0
        if n > args.offset:
            out_n = ((n - 1 - args.offset) // args.stride) + 1
        if args.expected_output and out_n != args.expected_output:
            raise ValueError(f"expected {args.expected_output} output rows, got {out_n}")

        args.output_bin.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output_bin.with_suffix(args.output_bin.suffix + ".tmp")
        with tmp.open("wb") as dst:
            dst.write(struct.pack("<II", out_n, dim))
            for out_idx in range(out_n):
                in_idx = args.offset + out_idx * args.stride
                src.seek(8 + in_idx * record_bytes)
                raw = src.read(record_bytes)
                if len(raw) != record_bytes:
                    raise ValueError(f"short read at input row {in_idx}")
                dst.write(raw)
        shutil.move(str(tmp), str(args.output_bin))
    print(f"wrote {args.output_bin} rows={out_n} dim={dim} stride={args.stride} offset={args.offset}")


if __name__ == "__main__":
    main()
