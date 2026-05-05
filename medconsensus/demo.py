from __future__ import annotations

import argparse
import json
from pathlib import Path

from medconsensus.orchestrator import MedConsensusOrchestrator
from medconsensus.renderer import render_markdown
from medconsensus.schemas import SyntheticCaseRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a MedConsensus demo report from an example JSON case.")
    parser.add_argument("case_file", type=Path, help="Path to a JSON file matching SyntheticCaseRequest.")
    args = parser.parse_args()

    request = SyntheticCaseRequest.model_validate(json.loads(args.case_file.read_text(encoding="utf-8")))
    report = MedConsensusOrchestrator().invoke(request)
    print(render_markdown(report))


if __name__ == "__main__":
    main()
