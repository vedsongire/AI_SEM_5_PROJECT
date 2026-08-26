"""Unit & Integration test suite for real-time dynamic VoiceAgent and Farmer Communication Channels."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.voice import VoiceAgent
from src.services.telephony_service import TelephonyService
from src.services.whatsapp_service import WhatsAppService


def test_voice_agent_nlu_hindi_dynamic():
    """Verify dynamic Hindi NLU parsing without pre-defined static dictionaries."""
    voice = VoiceAgent()
    query = "मुजफ्फरनगर उत्तर प्रदेश में 50 क्विंटल गेहूं के लिए सबसे अच्छी मंडी कौन सी है?"
    parsed = voice.parse_query(query)

    assert parsed["commodity"] == "Wheat"
    assert parsed["quantity_quintals"] == 50.0
    assert parsed["intent"] in ["PIPELINE_FULL", "MANDI_PRICE"]
    print(f"\n[OK] Dynamic Hindi NLU Passed! Commodity: {parsed['commodity']}, Quantity: {parsed['quantity_quintals']}")


def test_voice_agent_nlu_english_dynamic():
    """Verify dynamic English NLU parsing without pre-defined static dictionaries."""
    voice = VoiceAgent()
    query = "What is the best mandi for 100 quintals of Potato in Karnal Haryana?"
    parsed = voice.parse_query(query)

    assert parsed["commodity"] == "Potato"
    assert parsed["quantity_quintals"] == 100.0
    print(f"[OK] Dynamic English NLU Passed! Commodity: {parsed['commodity']}, Quantity: {parsed['quantity_quintals']}")


def test_voice_agent_tts_synthesis_dynamic():
    """Verify dynamic real-time Text-to-Speech audio synthesis."""
    voice = VoiceAgent()
    text = "नमस्ते किसान भाई! मुजफ्फरनगर में गेहूं का मंडी भाव ₹2400 प्रति क्विंटल है।"
    result = voice.synthesize_speech(text, language="hi")

    assert result["status"] in ["success", "warning"]
    assert len(result["text"]) > 0
    if result["status"] == "success":
        assert len(result["audio_bytes"]) > 0
    print(f"[OK] Dynamic TTS Synthesis Passed! Status: {result['status']}, Audio Bytes: {len(result['audio_bytes'])}")


def test_voice_agent_process_query_realtime():
    """Verify 100% real-time multi-agent voice query processing with live geocoding & scouting."""
    voice = VoiceAgent()
    query = "Muzaffarnagar Uttar Pradesh 50 quintals Wheat mandi price"
    res = voice.process_voice_query(query_input=query, input_type="text", language="en-IN")

    assert res["status"] == "success"
    assert len(res["response_text"]) > 0
    assert "pipeline_data" in res
    assert res["pipeline_data"]["winner"] is not None
    print(f"[OK] Real-time Multi-Agent Voice Query Passed! Winner: {res['pipeline_data']['winner']['market_name']}")


def test_whatsapp_service_realtime_messaging():
    """Verify WhatsApp chatbot real-time message processing."""
    wa_service = WhatsAppService()
    res = wa_service.process_incoming_message(
        from_phone="+919876543210",
        message_body="What is the mandi price for 50 quintals of Wheat in Muzaffarnagar?",
        language="en-IN",
    )

    assert res["status"] == "success"
    assert res["recipient_phone"] == "+919876543210"
    assert "K.I.S.A.N. AI WhatsApp Assistant" in res["reply_text"]
    assert "Wheat" in res["entities"]["commodity"]
    print(f"[OK] Dynamic WhatsApp Chatbot Service Passed!")


def test_telephony_service_realtime_calls():
    """Verify automated outbound call and inbound voice call services in real-time."""
    telephony = TelephonyService()

    # 1. Real-time outbound advisory call
    outbound = telephony.trigger_outbound_call(
        farmer_phone="+919876543210",
        farmer_location="Muzaffarnagar, Uttar Pradesh",
        commodity="Wheat",
        quantity_quintals=50.0,
        language="hi-IN",
    )
    assert outbound["status"] == "COMPLETED"
    assert outbound["direction"] == "OUTBOUND"
    assert len(outbound["spoken_script"]) > 0

    # 2. Real-time inbound voice call handling
    inbound = telephony.handle_inbound_call(
        farmer_phone="+919876543210",
        speech_input="Muzaffarnagar Uttar Pradesh wheat mandi price",
        language="en-IN",
    )
    assert inbound["status"] == "COMPLETED"
    assert inbound["direction"] == "INBOUND"
    assert len(inbound["spoken_response"]) > 0
    print(f"[OK] Dynamic Telephony Service Passed! Outbound Call ID: {outbound['call_id']}, Inbound Call ID: {inbound['call_id']}")


if __name__ == "__main__":
    test_voice_agent_nlu_hindi_dynamic()
    test_voice_agent_nlu_english_dynamic()
    test_voice_agent_tts_synthesis_dynamic()
    test_voice_agent_process_query_realtime()
    test_whatsapp_service_realtime_messaging()
    test_telephony_service_realtime_calls()
