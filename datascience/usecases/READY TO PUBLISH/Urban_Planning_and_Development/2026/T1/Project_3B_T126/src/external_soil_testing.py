"""
External Soil Testing - Project 3B T126

Tests whether external Canadian soil datasets can enrich the Kitchener water
main break records with useful soil or environmental attributes.

The workflow converts Kitchener break coordinates to longitude/latitude, then
runs spatial joins against two soil sources:
- Ontario Soil Survey Complex
- CANSIS / Soil Landscapes of Canada

Main use in notebook:
- run_external_soil_testing()
- display_soil_testing_summary()
"""

import pandas as pd
import geopandas as gpd
from pyproj import Transformer


ONTARIO_SOIL_KEEP_COLS = [
    "OGF_ID",
    "MAPUNIT",
    "SOIL_CMPLX",
    "PERCENT1",
    "SOIL_NAME1",
    "PARNT_MAT1",
    "SLOPE1",
    "DRAINAGE1",
    "DR_DESIGN1",
    "HYDRO1",
    "ATEXTURE1",
    "K_FACTOR1",
    "geometry",
]

SLC_POLYGON_KEEP_COLS = [
    "POLY_ID",
    "ECO_ID",
    "geometry",
]

CMP_KEEP_COLS = [
    "POLY_ID",
    "CMP",
    "PERCENT",
    "SLOPE",
    "STONE",
    "LOCSF",
    "PROVINCE",
    "SOIL_CODE",
    "MODIFIER",
    "PROFILE",
    "SOIL_ID",
    "CMP_ID",
]

SNT_KEEP_COLS = [
    "SOIL_ID",
    "SOILNAME",
    "DRAINAGE",
    "PMTEX1",
    "PMTEX2",
    "PMTEX3",
    "ORDER3",
    "G_GROUP3",
    "S_GROUP3",
]


def convert_break_coordinates(
    breaks_path,
    output_path=None,
    source_crs="EPSG:26917",
    target_crs="EPSG:4326",
):
    """Convert Kitchener break x/y coordinates into longitude/latitude."""
    breaks = pd.read_csv(breaks_path)
    breaks = breaks.dropna(subset=["x", "y"]).copy()

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    breaks["longitude"], breaks["latitude"] = transformer.transform(
        breaks["x"].values,
        breaks["y"].values,
    )

    if output_path:
        breaks.to_csv(output_path, index=False)

    return breaks


def create_break_points(breaks_latlon):
    """Create a GeoDataFrame of Kitchener break point locations."""
    return gpd.GeoDataFrame(
        breaks_latlon.copy(),
        geometry=gpd.points_from_xy(
            breaks_latlon["longitude"],
            breaks_latlon["latitude"],
        ),
        crs="EPSG:4326",
    )


def test_ontario_soil_join(breaks_latlon, soil_geojson_path):
    """Run spatial join against Ontario Soil Survey Complex data."""
    soil_gdf = gpd.read_file(soil_geojson_path)

    soil_small = soil_gdf[
        [col for col in ONTARIO_SOIL_KEEP_COLS if col in soil_gdf.columns]
    ].copy()

    break_points = create_break_points(breaks_latlon)

    joined = gpd.sjoin(
        break_points,
        soil_small,
        how="left",
        predicate="within",
    )

    summary = {
        "total_break_rows": len(joined),
        "matched_rows": int(joined["SOIL_NAME1"].notna().sum()),
        "match_pct": round(joined["SOIL_NAME1"].notna().mean() * 100, 2),
        "top_soil_names": joined["SOIL_NAME1"].value_counts(dropna=False).head(10),
        "top_texture_values_pct": (
            joined["ATEXTURE1"]
            .value_counts(normalize=True, dropna=False)
            .mul(100)
            .round(2)
            .head(10)
        ),
    }

    return joined, summary


def load_cansis_tables(slc_shp_path, cmp_path, snt_path):
    """Load CANSIS SLC polygon, component and soil name tables."""
    slc_gdf = gpd.read_file(slc_shp_path)
    cmp_df = pd.DataFrame(gpd.read_file(cmp_path))
    snt_df = pd.DataFrame(gpd.read_file(snt_path))

    return slc_gdf, cmp_df, snt_df


def clip_slc_to_kitchener_area(
    slc_gdf,
    bbox=(-80.65, -80.30, 43.35, 43.55),
):
    """Clip SLC polygons to approximate Kitchener-Waterloo area."""
    min_lon, max_lon, min_lat, max_lat = bbox
    return slc_gdf.cx[min_lon:max_lon, min_lat:max_lat].copy()


def select_dominant_soil_component(cmp_df):
    """Select the dominant soil component for each POLY_ID."""
    cmp_small = cmp_df[[c for c in CMP_KEEP_COLS if c in cmp_df.columns]].copy()
    cmp_small["PERCENT"] = pd.to_numeric(cmp_small["PERCENT"], errors="coerce")

    return (
        cmp_small
        .sort_values(["POLY_ID", "PERCENT"], ascending=[True, False])
        .drop_duplicates(subset=["POLY_ID"], keep="first")
        .copy()
    )


def test_cansis_slc_join(
    breaks_latlon,
    slc_shp_path,
    cmp_path,
    snt_path,
    bbox=(-80.65, -80.30, 43.35, 43.55),
):
    """Run spatial join against CANSIS / Soil Landscapes of Canada data."""
    slc_gdf, cmp_df, snt_df = load_cansis_tables(
        slc_shp_path=slc_shp_path,
        cmp_path=cmp_path,
        snt_path=snt_path,
    )

    slc_kw = clip_slc_to_kitchener_area(slc_gdf, bbox=bbox)
    slc_kw_small = slc_kw[
        [col for col in SLC_POLYGON_KEEP_COLS if col in slc_kw.columns]
    ].copy()

    break_points = create_break_points(breaks_latlon)
    slc_kw_small = slc_kw_small.to_crs(break_points.crs)

    breaks_slc = gpd.sjoin(
        break_points,
        slc_kw_small,
        how="left",
        predicate="within",
    )

    cmp_dominant = select_dominant_soil_component(cmp_df)

    joined = breaks_slc.merge(
        cmp_dominant,
        on="POLY_ID",
        how="left",
    )

    snt_small = snt_df[[c for c in SNT_KEEP_COLS if c in snt_df.columns]].copy()

    joined = joined.merge(
        snt_small,
        on="SOIL_ID",
        how="left",
    )

    summary = {
        "total_break_rows": len(joined),
        "matched_poly_rows": int(joined["POLY_ID"].notna().sum()),
        "unmatched_poly_rows": int(joined["POLY_ID"].isna().sum()),
        "match_pct": round(joined["POLY_ID"].notna().mean() * 100, 2),
        "top_soil_names": joined["SOILNAME"].value_counts(dropna=False).head(10),
        "drainage_distribution": joined["DRAINAGE"].value_counts(dropna=False),
        "texture_distribution": joined["PMTEX1"].value_counts(dropna=False),
        "slope_distribution": joined["SLOPE"].value_counts(dropna=False),
        "stone_distribution": joined["STONE"].value_counts(dropna=False),
    }

    return joined, summary


def run_external_soil_testing(
    breaks_path,
    ontario_soil_path,
    slc_shp_path,
    cmp_path,
    snt_path,
    output_latlon_path=None,
):
    """Run both external soil spatial join tests."""
    breaks_latlon = convert_break_coordinates(
        breaks_path=breaks_path,
        output_path=output_latlon_path,
    )

    ontario_joined, ontario_summary = test_ontario_soil_join(
        breaks_latlon=breaks_latlon,
        soil_geojson_path=ontario_soil_path,
    )

    cansis_joined, cansis_summary = test_cansis_slc_join(
        breaks_latlon=breaks_latlon,
        slc_shp_path=slc_shp_path,
        cmp_path=cmp_path,
        snt_path=snt_path,
    )

    return {
        "breaks_latlon": breaks_latlon,
        "ontario_joined": ontario_joined,
        "ontario_summary": ontario_summary,
        "cansis_joined": cansis_joined,
        "cansis_summary": cansis_summary,
    }


def display_soil_testing_summary(results):
    """Display key soil testing outputs for notebook demonstration."""
    ontario = results["ontario_summary"]
    cansis = results["cansis_summary"]

    print("Ontario Soil Survey Complex")
    print("Matched rows:", ontario["matched_rows"])
    print("Total rows:", ontario["total_break_rows"])
    print("Match %:", ontario["match_pct"])

    print("\nTop Ontario soil names:")
    display(ontario["top_soil_names"])

    print("\nTop Ontario texture values (%):")
    display(ontario["top_texture_values_pct"])

    print("\nCANSIS / Soil Landscapes of Canada")
    print("Matched polygon rows:", cansis["matched_poly_rows"])
    print("Unmatched polygon rows:", cansis["unmatched_poly_rows"])
    print("Match %:", cansis["match_pct"])

    print("\nTop CANSIS soil names:")
    display(cansis["top_soil_names"])

    print("\nCANSIS drainage distribution:")
    display(cansis["drainage_distribution"])

    print("\nCANSIS texture distribution:")
    display(cansis["texture_distribution"])