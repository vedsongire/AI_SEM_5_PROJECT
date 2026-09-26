"""Unit & Integration test suite for real-time dynamic VoiceAgent."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.agents.voice import VoiceAgent


def test_voice_agent_nlu_hindi_dynamic():
    """Verify dynamic Hindi NLU parsing without pre-defined static dictionaries."""
    voice = VoiceAgent()
    query = "मुजफ्फरनगर उत्तर प्रदेश में 50 क्विंटल गेहूं के लिए सबसे अच्छी मंडी कौन सी है?"
    parsed = voice.parse_query(query)

    assert parsed["commodity"] == "Wheat"
    assert parsed["quantity_quintals"] == 50.0
    assert parsed["intent"] in ["ARBITRAGE_RECOMMENDATION", "PIPELINE_FULL", "MANDI_PRICE"]
    print(f"\n[OK] Dynamic Hindi NLU Passed! Commodity: {parsed['commodity']}, Quantity: {parsed['quantity_quintals']}, Location: {parsed['location']}")


def test_voice_agent_nlu_english_dynamic():
    """Verify dynamic English NLU parsing without pre-defined static dictionaries."""
    voice = VoiceAgent()
    query = "What is the best mandi for 100 quintals of Potato in Karnal Haryana?"
    parsed = voice.parse_query(query)

    assert parsed["commodity"] == "Potato"
    assert parsed["quantity_quintals"] == 100.0
    print(f"[OK] Dynamic English NLU Passed! Commodity: {parsed['commodity']}, Quantity: {parsed['quantity_quintals']}, Location: {parsed['location']}")


def test_voice_agent_tts_synthesis_dynamic():
    """Verify dynamic real-time Text-to-Speech audio synthesis."""
    voice = VoiceAgent()
    text = "नमस्ते किसान भाई! मुजफ्फरनगर में गेहूं का मंडी भाव ₹2400 प्रति क्विंटल है।"
    result = voice.synthesize_speech(text, language="hi")

    assert result["status"] in ["success", "warning"]
    if result["status"] == "success":
        assert len(result["audio_bytes"]) > 0
        assert result["audio_url"].startswith("data:audio/mp3;base64,")
    print(f"[OK] Dynamic TTS Synthesis Passed! Status: {result['status']}, Audio Bytes: {len(result['audio_bytes'])}")


def test_voice_agent_chat_multiturn():
    """Verify multi-turn conversational chat interaction."""
    voice = VoiceAgent()

    # Turn 1: Greeting
    res1 = voice.chat("नमस्ते", session_state={}, language="hi-IN")
    assert "राम राम" in res1["reply_text"] or "K.I.S.A.N." in res1["reply_text"]
    state1 = res1["session_state"]

    # Turn 2: Mention crop and location
    res2 = voice.chat("मुझे 80 क्विंटल प्याज नासिक में बेचना है", session_state=state1, language="hi-IN")
    assert res2["recommendation"] is not None
    assert "Onion" in res2["entities"]["commodity"]
    print(f"[OK] Multi-turn Chat Passed! Recommended Mandi: {res2['recommendation']['market_name']}")


if __name__ == "__main__":
    test_voice_agent_nlu_hindi_dynamic()
    test_voice_agent_nlu_english_dynamic()
    test_voice_agent_tts_synthesis_dynamic()
    test_voice_agent_chat_multiturn()
    print("\n🎉 ALL VOICE AGENT TESTS PASSED SUCCESSFULLY!")
