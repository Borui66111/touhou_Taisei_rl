from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def test_policy_is_opt_in_and_input_reconciliation_precedes_gameplay_policy(patched_live_source):
    cli = (patched_live_source / "cli.c").read_text(encoding="utf-8")
    stage = (patched_live_source / "stage.c").read_text(encoding="utf-8")
    assert '"rl-policy"' in cli and "live_ppo_set_policy_path(optarg)" in cli
    assert "live_ppo_set_policy_path" not in cli.split("int cli_args", 1)[0]
    gameplay = stage[stage.index("static void stage_input(StageFrameState *fstate)"):stage.index("void stage_clear_hazards_predicate")]
    assert gameplay.index("events_poll") < gameplay.index("player_fix_input(&global.plr")
    assert gameplay.index("player_fix_input(&global.plr") < gameplay.index("live_ppo_apply_action()") < gameplay.index("player_applymovement(&global.plr)")


def test_policy_does_not_inject_during_pause_skip_transition_or_gameover(patched_live_source):
    stage = (patched_live_source / "stage.c").read_text(encoding="utf-8")
    gameplay = stage[stage.index("static void stage_input(StageFrameState *fstate)"):stage.index("void stage_clear_hazards_predicate")]
    assert "!fstate->paused && global.gameover==GAMEOVER_NONE && !stage_is_skip_mode() && !should_skip_frame(fstate)" in gameplay
    assert "stage_input_common" in stage and "KEY_STOP" in stage and "KEY_RESTART" in stage


def test_policy_maps_python_action_indices_including_neutral(patched_live_source):
    policy = (patched_live_source / "live_ppo.c").read_text(encoding="utf-8")
    expected = ("case 0:break;case 1:d|=INFLAG_UP;break;case 2:d|=INFLAG_DOWN;break;"
                "case 3:d|=INFLAG_LEFT;break;case 4:d|=INFLAG_RIGHT;break;"
                "case 5:d|=INFLAG_UP|INFLAG_LEFT;break;case 6:d|=INFLAG_UP|INFLAG_RIGHT;break;"
                "case 7:d|=INFLAG_DOWN|INFLAG_LEFT;break;case 8:d|=INFLAG_DOWN|INFLAG_RIGHT;break;")
    assert expected in policy and "if(a>=9)d|=INFLAG_FOCUS" in policy
    assert "uint32_t d=INFLAG_SHOT" in policy


def test_live_observation_matches_phase_timeout_and_laser_width_contract(patched_live_source):
    policy = (patched_live_source / "live_ppo.c").read_text(encoding="utf-8")
    assert "AT_Normal?1:a->type==AT_Move?2:a->type==AT_Spellcard?3:a->type==AT_SurvivalSpell?4:a->type==AT_ExtraSpell?5:0" in policy
    assert "a->starttime+a->timeout-global.frames" in policy
    assert "s->segment->width.b-s->segment->width.a)*s->segment_param)*0.5f" in policy
    assert "creal(p->pos)/480" in policy and "global.projs.first" in policy and "global.lasers.first" in policy


def test_ties_keep_native_spawn_order_and_binary_identity_is_verified(patched_live_source):
    policy = (patched_live_source / "live_ppo.c").read_text(encoding="utf-8")
    assert "return x->order<y->order?-1:x->order>y->order" in policy
    assert "sha256_digest(b,FILESIZE,digest,sizeof(digest))" in policy
    assert "0x47,0x6c,0xa4,0xe7" in policy
    assert "872d1e438afdc1df7858a0306c6373596d3ff89978b98d9f1b0a3d2aa378839d" in policy
    assert "RL policy binary SHA256 mismatch" in policy


def test_native_seed_decisions_and_outcome_are_logged(patched_live_source):
    stage = (patched_live_source / "stage.c").read_text(encoding="utf-8")
    policy = (patched_live_source / "live_ppo.c").read_text(encoding="utf-8")
    assert "live_ppo_start(seed, stage->id, global.diff" in stage and "live_ppo_finish(" in stage
    assert "RL run seed=%llu stage=%u difficulty=%d mode=%s practice=%d" in policy
    assert "RL decision frame=%d action=%d" in policy
    assert "RL policy outcome=%s frames=%d actions=%lu" in policy


def test_setup_script_uses_the_pinned_clone_for_build_and_resources():
    build_script = (ROOT / "scripts" / "build_taisei_sim.sh").read_text(encoding="utf-8")
    setup_script = (ROOT / "scripts" / "setup_ubuntu_wsl.sh").read_text(encoding="utf-8")
    pinned = "$ROOT/upstream/taisei-sim-pinned"
    assert 'ROOT_INPUT="${1:-$WORKBENCH_ROOT/upstream/taisei-sim-pinned}"' in build_script
    assert 'STAGING_CHECKOUT="$STAGE_DIR/checkout"' in build_script
    assert 'git clone --no-checkout "$REPO_URL" "$STAGING_CHECKOUT"' in build_script
    assert 'mv -T -n -- "$STAGING_CHECKOUT" "$ROOT"' in build_script
    assert "bash scripts/build_taisei_sim.sh" in setup_script
    assert f'find "{pinned}/build"' in setup_script
    assert f"TAISEI_NATIVE_RESOURCES='{pinned}/resources'" in setup_script
    assert f'meson devenv -C "{pinned}/build"' in setup_script
    assert f'--resource-path "{pinned}/resources"' in setup_script
