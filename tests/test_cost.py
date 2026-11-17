from datetime import date

import pytest

from remix import config, cost

TABLE = cost.load_table()
NOV = date(2026, 11, 17)
JAN = date(2027, 1, 1)


def test_price_follows_its_date_window():
    assert TABLE.price("flash-tts.audio-out", NOV)["usd"] == 9.0
    assert TABLE.price("flash-tts.audio-out", JAN)["usd"] == 18.0
    with pytest.raises(cost.CostError):
        TABLE.price("no-such-model", NOV)


def test_no_price_id_has_two_entries_on_the_same_day():
    for day in (NOV, date(2026, 12, 31), JAN):
        ids = [price["id"] for price in TABLE.current(day)]
        assert len(ids) == len(set(ids))


def test_music_is_per_song_and_a_song_covers_184_seconds():
    assert cost.total(cost.music(TABLE, NOV, 174)) == 0.08
    assert cost.total(cost.music(TABLE, NOV, 200)) == 0.16


def test_speech_is_25_tokens_a_second_and_doubles_in_2027():
    [line] = cost.speech(TABLE, NOV, 60)
    assert line.quantity == 1500 and line.usd == pytest.approx(0.0135)
    assert cost.total(cost.speech(TABLE, JAN, 60)) == pytest.approx(0.027)


def test_omni_video_prices_output_and_source_tokens():
    output, source = cost.omni_video(TABLE, NOV, 10, "720p", source_seconds=10)
    assert output.quantity == 57920 and output.usd == pytest.approx(0.695, abs=0.0005)
    assert source.usd == pytest.approx(0.0869, abs=0.0005)


def test_clip_length_limits_round_shots_up():
    assert cost.veo_seconds([2.0, 5.0, 9.0], TABLE) == 22  # 4 + 6 + two 6-second clips
    assert cost.omni_seconds([2.0, 5.2, 20.0], TABLE) == 29  # 3 + 6 + 20


def test_options_cover_every_mode_and_flag_the_cap(monkeypatch):
    monkeypatch.setattr(config, "CAP_USD", 5.0)
    options = cost.options_for(TABLE, NOV, 174.0, [3.0] * 58, 110.0, 1800)
    assert {option.mode for option in options} == {
        "audio remix", "re-voice / translate", "restyle", "regenerate", "motion-graphics rebuild",
    }
    priced = {(option.mode, option.name): option.usd for option in options}
    assert priced[("audio remix", "new music, original voices")] == 0.08
    assert cost.within_cap(priced[("re-voice / translate", "Gemini 3.8 Flash TTS")])
    assert not cost.within_cap(priced[("restyle", "Gemini Omni 1.2 Flash, 720p")])
    assert priced[("motion-graphics rebuild", "rendered from code, no provider")] == 0

    sheet = cost.render_options("demo", options, NOV)
    assert "| audio remix | new music, original voices | $0.08 |  |" in sheet
    assert "asks first" in sheet


def test_ledger_round_trip_and_rollups(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path)
    assert cost.entries() == []
    [song] = cost.music(TABLE, NOV, 60)
    [voice] = cost.speech(TABLE, NOV, 60)
    cost.record("demo", "first", song, estimated_usd=0.08)
    cost.record("demo", "first", voice)

    rows = cost.entries()
    assert [row["price_id"] for row in rows] == ["lyria-3-pro.song", "flash-tts.audio-out"]
    assert rows[0]["estimated_usd"] == 0.08
    assert cost.summarize(rows, "provider") == [("google", 0.0935, 2)]
    assert [key for key, _, _ in cost.summarize(rows, "model")] == ["gemini-3.8-flash-tts", "lyria-3-pro-preview"]
    assert cost.summarize(rows, "project") == [("demo", 0.0935, 2)]
    with pytest.raises(cost.CostError):
        cost.summarize(rows, "weather")


def test_credit_left_is_the_balance_minus_later_spend_with_that_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path)
    assert cost.credit_status([], NOV) is None
    cost.set_credit("welcome credit", "google", 100.0, ends=date(2027, 1, 15), as_of=NOV)
    rows = [
        {"provider": "google", "usd": 2.5, "at": "2026-11-18T10:00:00+00:00"},
        {"provider": "google", "usd": 9.0, "at": "2026-11-01T10:00:00+00:00"},  # before the balance was read
        {"provider": "other", "usd": 4.0, "at": "2026-11-18T10:00:00+00:00"},
    ]
    status = cost.credit_status(rows, date(2026, 12, 16))
    assert (status["spent"], status["left"], status["days_left"]) == (2.5, 97.5, 30)
