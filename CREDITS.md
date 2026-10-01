# Credits and licenses

BrainCog is developed by the BrainCog team and its upstream contributors.
Original project: https://github.com/BrainCog-X/Brain-Cog
Project website: http://www.brain-cog.network/
The original Git history, authorship, source notices, README, and research
citation are preserved in this fork. Refer to the upstream README for the full
author list and BibTeX entry for Zeng et al., *BrainCog*, Patterns (2023),
https://doi.org/10.1016/j.patter.2023.100789.

Kenneth Salmon (mcographics) maintains this fork and contributed the TanyaOS
integration: lifecycle-linked monitoring, multiscale diagnostic simulation,
software event routing, and standalone packaging. Adaptations were prepared
with assistance from OpenAI Codex. This fork is independent of the upstream
team; upstream endorsement is not implied.

## License inventory

| Scope | License file | Treatment |
| --- | --- | --- |
| Upstream BrainCog | [LICENSE](LICENSE) | Original Apache License 2.0 text retained byte for byte. Existing source notices remain. |
| Upstream MAToM-SNN example | [examples/Social_Cognition/MAToM-SNN/LICENSE](examples/Social_Cognition/MAToM-SNN/LICENSE) | Original GPL version 3 text retained byte for byte. The example retains its own terms. |
| New TanyaOS integration and associated fork documentation/tests | [tanyaos_braincog/LICENSE](tanyaos_braincog/LICENSE) | Apache License 2.0; copyright 2026 Kenneth Salmon. Source files identify their adaptation. |

No upstream NOTICE file was tracked at the fork's starting revision
`f9b879f`. No existing upstream license file or source attribution was removed.
Licenses in nested components continue to apply to their respective content;
the integration's license does not replace them.

The integration uses BrainCog, PyTorch, and psutil from the user's Python
environment. Their source and binary distributions are not bundled as new
vendored dependencies. Their own license notices remain in their respective
distributions. Upstream requirements and examples remain as provided upstream.
No model weights, voice packs, brain atlas assets, user databases, credentials,
or TanyaOS frontend assets are added by this integration.

The adapted TanyaOS files and their exact source revision and hashes are listed
in [PROVENANCE.json](tanyaos_braincog/PROVENANCE.json). Standalone changes replace
protected-core storage imports with explicit local storage, default discovery
to this fork, normalize/create the runtime directory, and allow an explicit
host storage provider. The event adapter
and multiscale algorithms are preserved from the recorded TanyaOS revision.
