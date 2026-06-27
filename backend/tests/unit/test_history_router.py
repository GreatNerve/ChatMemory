from app.services.history_router import fast_history_route


def test_fast_route_casual_short_replies():
    assert fast_history_route("lol") == "casual"
    assert fast_history_route("k") == "casual"
    assert fast_history_route("👍") == "casual"


def test_fast_route_obvious_memory():
    assert fast_history_route("yaad hai Goa trip kab plan hui thi?") == "memory"
    assert fast_history_route("what did we say about the meeting last time") == "memory"
    assert fast_history_route("us din kya bola tha") == "memory"


def test_fast_route_ambiguous():
    assert fast_history_route("so about that thing we discussed") == "ambiguous"
    assert fast_history_route("I was thinking about something") == "ambiguous"
