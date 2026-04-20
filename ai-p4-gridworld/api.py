import requests

from config import API_KEY, BASE_URL, OTP, TEAM_ID, USER_ID

HEADERS = {
    "userid": USER_ID,
    "x-api-key": API_KEY,
    "Content-Type": "application/x-www-form-urlencoded",
}


def _send(method, path, *, params=None, data=None, include_auth_headers=True):
    """Send an API request with a custom User-Agent (not python-requests)."""
    session = requests.Session()
    headers = HEADERS if include_auth_headers else None

    req = requests.Request(
        method=method,
        url=f"{BASE_URL}/{path}",
        headers=headers,
        params=params,
        data=data,
    )
    prepared = session.prepare_request(req)
    prepared.headers["User-Agent"] = "gridworld-client/1.0"

    try:
        response = session.send(prepared, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"code": "FAIL", "message": f"HTTP error: {exc}"}
    except ValueError:
        return {"code": "FAIL", "message": "Server returned non-JSON response."}


def get_location():
    return _send("GET", "gw.php", params={"type": "location", "teamId": TEAM_ID})


def enter_world(world_id):
    return _send(
        "POST",
        "gw.php",
        data={"type": "enter", "worldId": str(world_id), "teamId": TEAM_ID},
    )


def move(world_id, direction):
    return _send(
        "POST",
        "gw.php",
        data={
            "type": "move",
            "teamId": TEAM_ID,
            "worldId": str(world_id),
            "move": direction,
        },
    )


def reset():
    # reset.php uses teamId and otp in query string.
    return _send(
        "GET",
        "reset.php",
        params={"teamId": TEAM_ID, "otp": OTP},
        include_auth_headers=True,
    )