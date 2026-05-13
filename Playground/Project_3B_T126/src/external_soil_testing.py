import pandas as pd
import geopandas as gpd
from pyproj import Transformer


def convert_break_coordinates(
    breaks_path,
    output_path=None,
    source_crs="EPSG:26917",
    target_crs="EPSG:4326",
):
    """Convert Kitchener break x/y coordinates to longitude/latitude."""

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


def test_ontario_soil_join(breaks_latlon, soil_geojson_path):
    """Test Ontario Soil Survey Complex spatial join."""

    soil_gdf = gpd.read_file(soil_geojson_path)

    soil_keep_cols = [
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

    soil_small = soil_gdf[soil_keep_cols].copy()

    break_points = gpd.GeoDataFrame(
        breaks_latlon.copy(),
        geometry=gpd.points_from_xy(
            breaks_latlon["longitude"],
            breaks_latlon["latitude"],
        ),
        crs="EPSG:4326",
    )

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


def test_cansis_slc_join(
    breaks_latlon,
    slc_shp_path,
    cmp_path,
    snt_path,
    bbox=(-80.65, -80.30, 43.35, 43.55),
):
    """Test CANSIS / Soil Landscapes of Canada spatial join."""

    min_lon, max_lon, min_lat, max_lat = bbox

    slc_gdf = gpd.read_file(slc_shp_path)
    cmp_df = pd.DataFrame(gpd.read_file(cmp_path))
    snt_df = pd.DataFrame(gpd.read_file(snt_path))

    slc_kw = slc_gdf.cx[min_lon:max_lon, min_lat:max_lat].copy()

    break_points = gpd.GeoDataFrame(
        breaks_latlon.copy(),
        geometry=gpd.points_from_xy(
            breaks_latlon["longitude"],
            breaks_latlon["latitude"],
        ),
        crs="EPSG:4326",
    )

    slc_kw_small = slc_kw[["POLY_ID", "ECO_ID", "geometry"]].copy()
    slc_kw_small = slc_kw_small.to_crs(break_points.crs)

    breaks_slc = gpd.sjoin(
        break_points,
        slc_kw_small,
        how="left",
        predicate="within",
    )

    cmp_keep_cols = [
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

    cmp_small = cmp_df[[c for c in cmp_keep_cols if c in cmp_df.columns]].copy()
    cmp_small["PERCENT"] = pd.to_numeric(cmp_small["PERCENT"], errors="coerce")

    cmp_dominant = (
        cmp_small
        .sort_values(["POLY_ID", "PERCENT"], ascending=[True, False])
        .drop_duplicates(subset=["POLY_ID"], keep="first")
        .copy()
    )

    joined = breaks_slc.merge(cmp_dominant, on="POLY_ID", how="left")

    snt_keep_cols = [
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

    snt_small = snt_df[[c for c in snt_keep_cols if c in snt_df.columns]].copy()
    joined = joined.merge(snt_small, on="SOIL_ID", how="left")

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

    print("Ontario Soil Survey Complex")
    print("Matched rows:", results["ontario_summary"]["matched_rows"])
    print("Total rows:", results["ontario_summary"]["total_break_rows"])
    print("Match %:", results["ontario_summary"]["match_pct"])

    print("\nTop Ontario soil names:")
    display(results["ontario_summary"]["top_soil_names"])

    print("\nTop Ontario texture values (%):")
    display(results["ontario_summary"]["top_texture_values_pct"])

    print("\nCANSIS / Soil Landscapes of Canada")
    print("Matched polygon rows:", results["cansis_summary"]["matched_poly_rows"])
    print("Unmatched polygon rows:", results["cansis_summary"]["unmatched_poly_rows"])
    print("Match %:", results["cansis_summary"]["match_pct"])

    print("\nTop CANSIS soil names:")
    display(results["cansis_summary"]["top_soil_names"])

    print("\nCANSIS drainage distribution:")
    display(results["cansis_summary"]["drainage_distribution"])

    print("\nCANSIS texture distribution:")
    display(results["cansis_summary"]["texture_distribution"])