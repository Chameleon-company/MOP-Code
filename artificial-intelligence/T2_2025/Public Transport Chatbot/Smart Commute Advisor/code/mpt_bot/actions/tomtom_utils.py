# -------------------------------------------------------------------------------------------------
# Title: TomTom helpers (geocode + routing)
# Author: Juveria Nishath
# Purpose: Small, dependency-light utilities so the Rasa actions can call to:
#          1) turn a place name into (lat, lon, label)
#          2) get a driving route summary with live traffic
# -------------------------------------------------------------------------------------------------
import os, re, requests, urllib.parse
from typing import Optional, Tuple


def tt_geocode(query, api_key, country_set="AU", limit=1, bias_lat=None, bias_lon=None, radius_km=120) -> Optional[Tuple[float, float, str]]:
    """
    what it does:
      - takes whatever place/address text you give it,
      - asks TomTom Search for the best match,
      - hands back (lat, lon, nice_label) of the top hit.
    defaults:
      - country is AU so “Richmond” lands in VIC, not the US.
    returns:
      - (lat, lon, label) if it finds something, else None.
    """
    if not query:
        return None
    q = urllib.parse.quote(query)
    url = f"https://api.tomtom.com/search/2/search/{q}.json"
    params = {
        "key": api_key,
        "limit": limit,
        "countrySet": country_set,
        "idxSet": "POI,Geo",
    }
    if bias_lat is not None and bias_lon is not None:
        params["lat"] = f"{bias_lat:.6f}"
        params["lon"] = f"{bias_lon:.6f}"
        params["radius"] = int(radius_km * 1000)  # meters

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()
    results = j.get("results") or []
    if not results:
        return None

    top = results[0]
    pos = top["position"]
    label = (top.get("address", {}) or {}).get("freeformAddress") or query
    return float(pos["lat"]), float(pos["lon"]), label


def tt_route(o_lat: float, o_lon: float, d_lat: float, d_lon: float, api_key: str) -> Optional[dict]:
    """
    what it does:
      - asks TomTom Routing for the fastest *car* route right now (traffic on),
      - gives you the summary: ETA (with and without traffic), traffic delay, distance.
    returns:
      - summary dict for the first route, or None if no route.
    """
    locs = f"{o_lat:.6f},{o_lon:.6f}:{d_lat:.6f},{d_lon:.6f}"
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{locs}/json"
    params = {
        "key": api_key,
        "traffic": "true",
        "routeType": "fastest",
        "travelMode": "car",           # explicit
        "computeBestOrder": "false",
        "computeTravelTimeFor": "all",
    }
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    routes = data.get("routes")
    if not routes:
        return None
    return routes[0].get("summary", {})


def fmt_time(sec: Optional[float]) -> Optional[str]:
    """
    what it does:
      - turns raw seconds into human time like "23 min" or "1 hr 12 min".
      - if you hand it None/bad data, it returns None instead of blowing up.
    """
    if sec is None:
        return None
    mins = int(round(float(sec) / 60.0))
    if mins < 60:
        return f"{mins} min"
    h, m = divmod(mins, 60)
    return f"{h} hr {m} min" if m else f"{h} hr"


def fmt_km(m: Optional[float]) -> Optional[str]:
    """
    what it does:
      - converts meters to a neat "x.y km" string.
      - returns None if you give it nothing.
    """
    return f"{float(m) / 1000.0:.1f} km" if m is not None else None
