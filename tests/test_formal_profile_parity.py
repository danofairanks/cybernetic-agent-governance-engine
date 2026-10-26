# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Production stage selection must equal the formal model's, tier by tier.

The proof (``proof/model.py``) and the pipeline
(``src/gateway/governance/governor/pipeline.py``) each define which stages run
under which profile. Neither imports the other; this test is the bridge.
"""

import pytest

from proof.model import (
    JURISDICTION_TIER_PHASE,
    PLUGIN_TIER_PHASE,
    POST_HITL_READ_ONLY_TIERS,
    PROFILES,
    REVALIDATED_TIERS,
    TIER_PHASE,
    TIERS,
    WARRANT_TIER,
    WARRANT_TIER_PHASE,
    post_hitl_rechecks_warrant,
    post_hitl_runs_every_phase2_tier,
    runs_under_profile,
)
from proof.model import (
    PROFILE_STAGES as PROOF_PROFILES,
)
from src.gateway.governance.governor.pipeline import (
    POST_HITL_READ_ONLY_STAGES,
    Profile,
    stage_runs_under,
)

pytestmark = [pytest.mark.unit, pytest.mark.local]

_ALL_PHASES = (
    TIER_PHASE | PLUGIN_TIER_PHASE | WARRANT_TIER_PHASE | JURISDICTION_TIER_PHASE
)


def test_profile_names_match() -> None:
    assert {p.value for p in Profile} == set(PROFILES)


@pytest.mark.parametrize("profile", list(Profile))
@pytest.mark.parametrize(("tier", "phase"), sorted(_ALL_PHASES.items()))
def test_production_predicate_matches_proof(
    profile: Profile, tier: str, phase: int
) -> None:
    """Every (profile, tier, phase) selects identically: kernel, plugin-named
    or jurisdiction tier. The runtime reads a contributed tier's declaration
    (``revalidated``), the model its name in REVALIDATED_TIERS; the
    declarations themselves are pinned by
    ``test_region_tiers_match_proof`` and ``test_jurisdiction_wiring``."""
    assert stage_runs_under(
        profile,
        name=tier,
        mutating=phase == 2,
        revalidated=tier in REVALIDATED_TIERS,
    ) == runs_under_profile(profile.value, tier, phase)


def test_full_and_dry_run_run_every_tier() -> None:
    assert PROOF_PROFILES["FULL"] == frozenset(TIERS)
    assert PROOF_PROFILES["DRY_RUN"] == frozenset(TIERS)


def test_post_hitl_read_only_sets_agree_and_are_proof_tiers() -> None:
    """Every re-checked read-only stage is a modelled tier: a TIERS member or
    the named warrant tier (which adds no states; see proof/model.py)."""
    assert POST_HITL_READ_ONLY_STAGES == POST_HITL_READ_ONLY_TIERS
    assert POST_HITL_READ_ONLY_STAGES <= frozenset(TIERS) | frozenset(
        WARRANT_TIER_PHASE
    )


def test_every_jurisdiction_tier_is_revalidated_in_the_model() -> None:
    """A jurisdiction assessment can change while a human decides: the model
    revalidates every jurisdiction tier (and only read-only tiers)."""
    assert frozenset(JURISDICTION_TIER_PHASE) <= REVALIDATED_TIERS
    assert all(_ALL_PHASES[t] == 1 for t in REVALIDATED_TIERS)


def test_warrant_tier_is_the_kernel_warrant_stage() -> None:
    from src.gateway.governance.governor.stages.warrant import WarrantStage

    assert WarrantStage.name == WARRANT_TIER
    assert WarrantStage.mutating is (WARRANT_TIER_PHASE[WARRANT_TIER] == 2)
    assert stage_runs_under(
        Profile.POST_HITL, name=WarrantStage.name, mutating=False, revalidated=False
    )
    assert post_hitl_rechecks_warrant()


def test_post_hitl_kernel_scope_is_opa_plus_phase2() -> None:
    assert PROOF_PROFILES["POST_HITL"] == frozenset({"opa", "cbf", "fiscal"})


def test_post_hitl_claim_holds() -> None:
    assert post_hitl_runs_every_phase2_tier()


def test_post_hitl_skips_read_only_plugin_tiers() -> None:
    """Fail-closed scope, not fail-open: a phase-1 plugin tier that does not
    declare ``revalidated_after_approval`` is not re-run."""
    assert not stage_runs_under(
        Profile.POST_HITL, name="clinical_consensus", mutating=False, revalidated=False
    )


def test_post_hitl_reruns_a_declared_tier_whatever_its_name() -> None:
    """Structural, never by name: any read-only tier that declares
    ``revalidated_after_approval`` re-runs after approval."""
    assert stage_runs_under(
        Profile.POST_HITL, name="any_contributed_name", mutating=False, revalidated=True
    )


# ── Jurisdiction tiers, once per region (D-L, proof claim 10) ──────────────────

from proof.model import (  # noqa: E402
    JURISDICTION_TIERS,
    post_hitl_rechecks_jurisdiction,
    region_profile_stages,
    region_tier_phase,
)
from src.gateway.governance.jurisdiction import (  # noqa: E402
    JURISDICTIONS,
    resolve_jurisdiction,
)


def test_proof_and_production_know_the_same_regions() -> None:
    assert set(JURISDICTION_TIERS) == set(JURISDICTIONS)


@pytest.mark.parametrize("region", sorted(JURISDICTION_TIERS))
def test_region_tiers_match_proof(region: str) -> None:
    """The production contribution adds exactly the proof's tiers, in the
    proof's phase, and every one declares ``revalidated_after_approval`` as
    the model's REVALIDATED_TIERS says (GHSA-f6xq-23f7-9grm)."""
    contribution = resolve_jurisdiction(region)
    phases = region_tier_phase(region)
    assert tuple(t.tier_name for t in contribution.tiers) == JURISDICTION_TIERS[region]
    assert {t.tier_name: t.phase for t in contribution.tiers} == {
        t: phases[t] for t in JURISDICTION_TIERS[region]
    }
    assert {t.tier_name: t.revalidated_after_approval for t in contribution.tiers} == {
        t: t in REVALIDATED_TIERS for t in JURISDICTION_TIERS[region]
    }


@pytest.mark.parametrize("profile", list(Profile))
@pytest.mark.parametrize("region", sorted(JURISDICTION_TIERS))
def test_region_stage_selection_matches_proof(region: str, profile: Profile) -> None:
    """Per region, the pipeline predicate selects the proof's profile set."""
    phases = region_tier_phase(region)
    selected = frozenset(
        tier
        for tier, phase in phases.items()
        if stage_runs_under(
            profile,
            name=tier,
            mutating=phase == 2,
            revalidated=tier in REVALIDATED_TIERS,
        )
    )
    assert selected == region_profile_stages(region)[profile.value]


def test_jurisdiction_claim_holds() -> None:
    assert post_hitl_rechecks_jurisdiction()


# ── Phase-2 gate and the pending-approval outcome (proof claims 8 and 9) ───────

from proof.model import (  # noqa: E402
    VIOLATION_KINDS,
    _kind_sets,
    awaits_approval,
    hard_preview_denies_before_hitl,
    no_commit_under_pending_findings,
    pending_approval_outcome,
)
from proof.model import (
    phase2_mode as proof_phase2_mode,
)
from src.gateway.governance.classification_engine import (  # noqa: E402
    ClassificationContext,
    ClassificationEngine,
)
from src.gateway.governance.contracts import Violation, ViolationKind  # noqa: E402
from src.gateway.governance.decisions import GovernanceDecision  # noqa: E402
from src.gateway.governance.governor.pipeline import (  # noqa: E402
    StageContext,
    phase2_mode,
    run_pipeline,
)
from src.gateway.governance.governor.reservation import ReservationScope  # noqa: E402
from src.gateway.governance.narrower import NarrowerRegistry  # noqa: E402

_KIND_SETS = list(_kind_sets())
_PENDING = [k for k in _KIND_SETS if awaits_approval(k)]


def _violations(tier: str, kinds: frozenset[str]) -> list[Violation]:
    return [
        Violation(
            tier=tier, code=f"{tier.upper()}_{k}", message=k, kind=ViolationKind[k]
        )
        for k in sorted(kinds)
    ]


class _Phase1Probe:
    """A read-only stage that reports exactly the given kinds.

    Named ``opa`` because OPA is the read-only stage every profile runs,
    POST_HITL included (``POST_HITL_READ_ONLY_STAGES``).
    """

    name, mutating = "opa", False

    def __init__(self, kinds: frozenset[str]) -> None:
        self._kinds = kinds

    async def run(self, ctx: StageContext) -> list[Violation]:
        return _violations(self.name, self._kinds)


class _BarrierProbe:
    """A claiming phase-2 tier whose preview reports the given kinds; never commits."""

    name, mutating = "barrier_probe", True

    def __init__(self, kinds: frozenset[str]) -> None:
        self._kinds = kinds

    def claims(self, ctx: StageContext) -> bool:
        return True

    async def preview(self, ctx: StageContext) -> list[Violation]:
        return _violations(self.name, self._kinds)

    async def commit(self, ctx: StageContext):  # pragma: no cover - must never run
        raise AssertionError("phase 2 committed while approval was pending")

    async def rollback(self, ctx: StageContext, receipt) -> None:  # pragma: no cover
        raise AssertionError("nothing was committed")


def test_violation_kinds_match() -> None:
    assert {k.name for k in ViolationKind} == set(VIOLATION_KINDS)


@pytest.mark.parametrize("profile", list(Profile))
@pytest.mark.parametrize(
    "kinds", _KIND_SETS, ids=lambda k: "+".join(sorted(k)) or "clean"
)
def test_production_phase2_gate_matches_proof(
    profile: Profile, kinds: frozenset[str]
) -> None:
    assert phase2_mode(
        profile, [ViolationKind[k] for k in kinds]
    ).value == proof_phase2_mode(profile.value, kinds)


def test_phase2_claims_hold() -> None:
    assert no_commit_under_pending_findings()
    assert hard_preview_denies_before_hitl()


@pytest.mark.parametrize("profile", [Profile.DRY_RUN, Profile.FULL, Profile.POST_HITL])
@pytest.mark.parametrize("phase1", _PENDING, ids=lambda k: "+".join(sorted(k)))
@pytest.mark.parametrize(
    "preview", _KIND_SETS, ids=lambda k: "+".join(sorted(k)) or "pass"
)
async def test_pending_approval_outcome_matches_proof(
    profile: Profile, phase1: frozenset[str], preview: frozenset[str]
) -> None:
    """The real pipeline + classifier decide what the model decides, and commit nothing."""
    stages = [_Phase1Probe(phase1), _BarrierProbe(preview)]
    ctx = StageContext(action="act", params={}, profile=profile)
    if profile == Profile.DRY_RUN:
        result = await run_pipeline(stages, ctx, profile=profile)
    else:
        async with ReservationScope() as scope:
            result = await run_pipeline(stages, ctx, profile=profile, scope=scope)
        assert result.commits == () and result.committed_stages == ()

    classification = ClassificationEngine(NarrowerRegistry(narrowers=[])).classify(
        ClassificationContext(
            violations=list(result.violations),
            confidence=1.0,
            opa_decision=None,
            policy_ambiguous=False,
            params={},
        ),
        "act",
    )
    decided = {GovernanceDecision.DENY: "DENIED"}.get(
        classification.decision, classification.decision.value
    )
    assert (decided, result.barrier_preview.value) == pending_approval_outcome(
        phase1, preview
    )


# ── After approval: only the approved holds are answered (proof claim 14) ─────

from typing import Any  # noqa: E402

from proof.model import (  # noqa: E402
    approval_answers_only_its_holds,
    post_hitl_findings,
    post_hitl_verdict,
)
from src.gateway.governance.contracts import (  # noqa: E402
    CommitReceipt,
    MutatingTier,
    ReadOnlyTier,
)
from src.gateway.governance.governor.pipeline import answered_by_approval  # noqa: E402
from src.gateway.governance.governor.stages.domain_tiers import (  # noqa: E402
    order_stages,
)

_PROBE = "revalidated_probe"
_PROBE_HOLD = (_PROBE, f"{_PROBE.upper()}_HITL")


class _RevalidatedProbe(ReadOnlyTier):
    """A contributed read-only tier that declares revalidation, reporting ``kinds``."""

    tier_name = _PROBE
    order = 1
    revalidated_after_approval = True

    def __init__(self, kinds: frozenset[str]) -> None:
        self._kinds = kinds

    def claims_action(self, action: str, params: dict[str, Any]) -> bool:
        return True

    async def evaluate(self, action: str, params: dict[str, Any]) -> list[Violation]:
        return _violations(self.tier_name, self._kinds)


class _CommittingBarrier(MutatingTier):
    """A claiming phase-2 tier that always commits."""

    tier_name = "committing_barrier"
    order = 1

    def claims_action(self, action: str, params: dict[str, Any]) -> bool:
        return True

    async def evaluate(self, action: str, params: dict[str, Any]) -> list[Violation]:
        return []

    async def commit(
        self, action: str, params: dict[str, Any]
    ) -> tuple[list[Violation], CommitReceipt | None]:
        return [], CommitReceipt(tier=self.tier_name, magnitude=1.0)

    async def rollback(
        self, action: str, params: dict[str, Any], receipt: CommitReceipt
    ) -> None:
        return None

    async def confirm(
        self, action: str, params: dict[str, Any], receipt: CommitReceipt
    ) -> None:
        return None


def test_approval_holds_claim_holds() -> None:
    assert approval_answers_only_its_holds()


@pytest.mark.parametrize("approved", [False, True], ids=["unapproved", "approved"])
@pytest.mark.parametrize(
    "kinds", _KIND_SETS, ids=lambda k: "+".join(sorted(k)) or "clean"
)
async def test_post_hitl_answers_what_the_proof_answers(
    kinds: frozenset[str], approved: bool
) -> None:
    """The real pipeline re-runs a declared tier under POST_HITL, answers
    exactly the holds the approval covers, and commits iff the model seals."""
    stages = order_stages([_RevalidatedProbe(kinds), _CommittingBarrier()])
    ctx = StageContext(
        action="act",
        params={},
        profile=Profile.POST_HITL,
        approved_holds=frozenset({_PROBE_HOLD}) if approved else frozenset(),
    )
    async with ReservationScope() as scope:
        result = await run_pipeline(stages, ctx, profile=Profile.POST_HITL, scope=scope)

    remaining = post_hitl_findings(kinds, approved=approved)
    assert _PROBE in dict(result.plan)
    assert frozenset(v.kind.name for v in result.violations) == remaining
    sealable = post_hitl_verdict(remaining) == "ALLOW"
    assert result.committed_stages == (("committing_barrier",) if sealable else ())


@pytest.mark.parametrize("profile", [Profile.FULL, Profile.DRY_RUN])
def test_an_approval_answers_nothing_outside_post_hitl(profile: Profile) -> None:
    hold = Violation(
        tier=_PROBE_HOLD[0],
        code=_PROBE_HOLD[1],
        message="hold",
        kind=ViolationKind.HITL,
    )
    ctx = StageContext(
        action="act",
        params={},
        profile=profile,
        approved_holds=frozenset({_PROBE_HOLD}),
    )
    assert not answered_by_approval(ctx, hold)


@pytest.mark.parametrize("kind", [k for k in ViolationKind if k != ViolationKind.HITL])
def test_an_approval_answers_no_kind_but_hitl(kind: ViolationKind) -> None:
    """Matching (tier, code) is not enough: only a HITL finding is answered."""
    finding = Violation(
        tier=_PROBE_HOLD[0], code=_PROBE_HOLD[1], message="x", kind=kind
    )
    ctx = StageContext(
        action="act",
        params={},
        profile=Profile.POST_HITL,
        approved_holds=frozenset({_PROBE_HOLD}),
    )
    assert not answered_by_approval(ctx, finding)


class _StaticPolicy:
    """A policy client answering one verdict for every input."""

    def __init__(self, verdict: str) -> None:
        self._verdict = verdict

    async def evaluate_policy(self, input_data: dict[str, Any], **kwargs: Any) -> str:
        return self._verdict


_OPA_HOLD = ("governance", "OPA_MANUAL_REVIEW")


@pytest.mark.parametrize("approved", [False, True], ids=["unapproved", "approved"])
async def test_post_hitl_answers_opa_manual_review_only_when_approved(
    approved: bool,
) -> None:
    """OPA is a re-run read-only stage like any other: its MANUAL_REVIEW hold
    is answered iff the approval was given against it, and phase 2 commits
    iff the model seals (``post_hitl_findings({"HITL"}, approved=...)``)."""
    from src.gateway.governance.governor.stages.opa import OpaStage

    stages = (
        OpaStage(_StaticPolicy("MANUAL_REVIEW")),
        *order_stages([_CommittingBarrier()]),
    )
    ctx = StageContext(
        action="act",
        params={},
        profile=Profile.POST_HITL,
        approved_holds=frozenset({_OPA_HOLD}) if approved else frozenset(),
    )
    async with ReservationScope() as scope:
        result = await run_pipeline(stages, ctx, profile=Profile.POST_HITL, scope=scope)

    remaining = post_hitl_findings(frozenset({"HITL"}), approved=approved)
    assert frozenset(v.kind.name for v in result.violations) == remaining
    assert [v.code for v in result.violations] == (
        [] if approved else ["OPA_MANUAL_REVIEW"]
    )
    sealable = post_hitl_verdict(remaining) == "ALLOW"
    assert result.committed_stages == (("committing_barrier",) if sealable else ())
