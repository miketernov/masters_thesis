import os
import math
import numpy as np
import pandas as pd
import folium

from sklearn.cluster import KMeans
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import transform
from pyproj import Transformer
from scipy.spatial import Voronoi

# =========================
# 1. PATHS
# =========================
INPUT_FILE = r"xxx"
OUTPUT_DIR = r"xxx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. SETTINGS
# =========================
TARGET_COL = "target"
REST_TYPE_COL = "rest_type"

# если 1 = закрытие
TARGET_MODE = "closure"

# если 1 = выживание
# TARGET_MODE = "success"

N_POLYGONS = 50
MIN_TOTAL_OBS_IN_POLYGON = 3
MIN_OBS_PER_TYPE_IN_POLYGON = 3

GRID_STEP_M = 160

# =========================
# 3. APPROXIMATE SADOVOE POLYGON
# =========================
# (lat, lon)
SADOVOE_POLYGON_LATLON = [
    (55.7715, 37.5840),
    (55.7765, 37.5965),
    (55.7815, 37.6115),
    (55.7835, 37.6285),
    (55.7805, 37.6465),
    (55.7720, 37.6645),
    (55.7600, 37.6775),
    (55.7465, 37.6810),
    (55.7345, 37.6760),
    (55.7245, 37.6665),
    (55.7180, 37.6525),
    (55.7145, 37.6360),
    (55.7135, 37.6200),
    (55.7145, 37.6025),
    (55.7185, 37.5875),
    (55.7260, 37.5745),
    (55.7365, 37.5665),
    (55.7485, 37.5625),
    (55.7605, 37.5650),
    (55.7685, 37.5725),
]

# =========================
# 4. LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

required_cols = {TARGET_COL, REST_TYPE_COL}
missing_required = required_cols - set(df.columns)
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

# =========================
# 5. PREPARE COORDINATES
# =========================
if "latitude" in df.columns and "longitude" in df.columns:
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
elif "coordinates" in df.columns:
    coords = df["coordinates"].astype(str).str.split(",", expand=True)
    if coords.shape[1] != 2:
        raise ValueError("Column 'coordinates' must contain values like '55.75,37.61'")
    df["latitude"] = pd.to_numeric(coords[0].str.strip(), errors="coerce")
    df["longitude"] = pd.to_numeric(coords[1].str.strip(), errors="coerce")
else:
    raise ValueError("Need either ['latitude', 'longitude'] or ['coordinates'] columns")

df = df.dropna(subset=["latitude", "longitude", TARGET_COL, REST_TYPE_COL]).copy()

# =========================
# 6. SCORE
# =========================
if TARGET_MODE == "closure":
    # выше = лучше, т.к. ниже доля закрытий
    df["score"] = 1 - df[TARGET_COL].astype(float)
elif TARGET_MODE == "success":
    # выше = лучше, т.к. выше доля выживания
    df["score"] = df[TARGET_COL].astype(float)
else:
    raise ValueError("TARGET_MODE must be either 'closure' or 'success'")

# =========================
# 7. PROJECTIONS
# =========================
to_xy = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

def lonlat_to_xy(lon, lat):
    return to_xy.transform(lon, lat)

def xy_to_lonlat(x, y):
    return to_ll.transform(x, y)

polygon_ll = Polygon([(lon, lat) for lat, lon in SADOVOE_POLYGON_LATLON])
polygon_xy = transform(lambda x, y: to_xy.transform(x, y), polygon_ll)

# =========================
# 8. FILTER OBJECTS INSIDE POLYGON
# =========================
xy_coords = df.apply(lambda r: lonlat_to_xy(r["longitude"], r["latitude"]), axis=1)
df["x"] = [p[0] for p in xy_coords]
df["y"] = [p[1] for p in xy_coords]

df["inside_polygon"] = df.apply(
    lambda r: polygon_xy.contains(Point(r["x"], r["y"])) or polygon_xy.touches(Point(r["x"], r["y"])),
    axis=1
)

df_in = df[df["inside_polygon"]].copy()

if df_in.empty:
    raise ValueError("No observations fell inside the Sadovoe polygon.")

# =========================
# 9. GENERATE CANDIDATE POINTS INSIDE POLYGON
# =========================
minx, miny, maxx, maxy = polygon_xy.bounds
grid_points = []

x_vals = np.arange(minx, maxx + GRID_STEP_M, GRID_STEP_M)
y_vals = np.arange(miny, maxy + GRID_STEP_M, GRID_STEP_M)

for x in x_vals:
    for y in y_vals:
        pt = Point(x, y)
        if polygon_xy.contains(pt):
            grid_points.append([x, y])

grid_points = np.array(grid_points)

if len(grid_points) < N_POLYGONS:
    raise ValueError(
        f"Too few grid points inside polygon ({len(grid_points)}). Reduce GRID_STEP_M."
    )

# =========================
# 10. SELECT 20 CENTERS
# =========================
kmeans = KMeans(n_clusters=N_POLYGONS, random_state=42, n_init=20)
kmeans.fit(grid_points)
centers_xy = kmeans.cluster_centers_

# =========================
# 11. VORONOI HELPER
# =========================
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.
    Source adapted from SciPy cookbook.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

# =========================
# 12. BUILD CLIPPED VORONOI POLYGONS
# =========================
vor = Voronoi(centers_xy)
regions, vertices = voronoi_finite_polygons_2d(vor, radius=1_000_000)

zone_rows = []

for zone_id, region in enumerate(regions, start=1):
    poly_coords = vertices[region]
    poly = Polygon(poly_coords)

    if not poly.is_valid:
        poly = poly.buffer(0)

    clipped = poly.intersection(polygon_xy)

    if clipped.is_empty:
        continue

    if isinstance(clipped, MultiPolygon):
        # берем крупнейшую часть
        clipped = max(clipped.geoms, key=lambda g: g.area)

    center_x, center_y = centers_xy[zone_id - 1]
    center_lon, center_lat = xy_to_lonlat(center_x, center_y)

    zone_rows.append({
        "polygon_id": zone_id,
        "center_x": center_x,
        "center_y": center_y,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "zone_area_m2": clipped.area,
        "geometry": clipped
    })

zones_df = pd.DataFrame(zone_rows)

if len(zones_df) != N_POLYGONS:
    print(f"Warning: expected {N_POLYGONS} zones, got {len(zones_df)}")

# =========================
# 13. ASSIGN OBJECTS TO POLYGONS
# =========================
polygon_geoms = dict(zip(zones_df["polygon_id"], zones_df["geometry"]))

def assign_polygon_id(x, y):
    pt = Point(x, y)
    for pid, geom in polygon_geoms.items():
        if geom.contains(pt) or geom.touches(pt):
            return pid

    # fallback: nearest center
    dists = [
        ((x - row["center_x"]) ** 2 + (y - row["center_y"]) ** 2, row["polygon_id"])
        for _, row in zones_df.iterrows()
    ]
    return min(dists, key=lambda t: t[0])[1]

df_in["polygon_id"] = df_in.apply(lambda r: assign_polygon_id(r["x"], r["y"]), axis=1)

# =========================
# 14. AGGREGATE BY POLYGON AND TYPE
# =========================
polygon_type_stats = (
    df_in.groupby(["polygon_id", REST_TYPE_COL], as_index=False)
         .agg(
             n=("score", "size"),
             mean_score=("score", "mean"),
             mean_target=(TARGET_COL, "mean")
         )
)

polygon_type_stats = polygon_type_stats[
    polygon_type_stats["n"] >= MIN_OBS_PER_TYPE_IN_POLYGON
].copy()

polygon_overall_stats = (
    df_in.groupby("polygon_id", as_index=False)
         .agg(
             total_n=("score", "size"),
             overall_score=("score", "mean")
         )
)

ranked = polygon_type_stats.sort_values(
    ["polygon_id", "mean_score", "n"],
    ascending=[True, False, False]
).copy()

ranked["rank"] = ranked.groupby("polygon_id").cumcount() + 1

best_by_polygon = ranked[ranked["rank"] == 1].copy().rename(columns={
    REST_TYPE_COL: "best_type",
    "n": "best_type_n",
    "mean_score": "best_score",
    "mean_target": "best_mean_target"
})

second_by_polygon = ranked[ranked["rank"] == 2].copy().rename(columns={
    REST_TYPE_COL: "second_type",
    "n": "second_type_n",
    "mean_score": "second_score",
    "mean_target": "second_mean_target"
})

polygon_summary = zones_df.drop(columns=["geometry"]).merge(
    polygon_overall_stats, on="polygon_id", how="left"
).merge(
    best_by_polygon[["polygon_id", "best_type", "best_type_n", "best_score", "best_mean_target"]],
    on="polygon_id",
    how="left"
).merge(
    second_by_polygon[["polygon_id", "second_type", "second_type_n", "second_score", "second_mean_target"]],
    on="polygon_id",
    how="left"
)

polygon_summary["gap_1st_2nd"] = polygon_summary["best_score"] - polygon_summary["second_score"]

def get_status(row):
    total_n = row["total_n"]
    if pd.isna(total_n) or total_n < MIN_TOTAL_OBS_IN_POLYGON:
        return "insufficient_data"
    if pd.isna(row["best_type"]):
        return "no_type_passed_threshold"
    return "ok"

polygon_summary["status"] = polygon_summary.apply(get_status, axis=1)

# =========================
# 15. TOP-3 TYPES STRING
# =========================
top3_rows = ranked[ranked["rank"] <= 3].copy()
top3_rows["top_str"] = (
    top3_rows["rank"].astype(str) + ") " +
    top3_rows[REST_TYPE_COL].astype(str) +
    " | score=" + top3_rows["mean_score"].round(3).astype(str) +
    " | n=" + top3_rows["n"].astype(str)
)

top3_summary = (
    top3_rows.groupby("polygon_id")["top_str"]
    .apply(lambda s: "<br>".join(s.tolist()))
    .reset_index()
    .rename(columns={"top_str": "top_3_types"})
)

polygon_summary = polygon_summary.merge(top3_summary, on="polygon_id", how="left")

# =========================
# 16. SAVE TABLES
# =========================
zones_meta_to_save = zones_df.drop(columns=["geometry"]).copy()
zones_meta_to_save.to_csv(os.path.join(OUTPUT_DIR, "polygons_meta.csv"), index=False)
polygon_type_stats.to_csv(os.path.join(OUTPUT_DIR, "polygon_type_stats.csv"), index=False)
polygon_summary.to_csv(os.path.join(OUTPUT_DIR, "polygon_summary.csv"), index=False)
df_in.to_csv(os.path.join(OUTPUT_DIR, "objects_assigned_to_polygons.csv"), index=False)

print("Saved CSV files:")
print(os.path.join(OUTPUT_DIR, "polygons_meta.csv"))
print(os.path.join(OUTPUT_DIR, "polygon_type_stats.csv"))
print(os.path.join(OUTPUT_DIR, "polygon_summary.csv"))
print(os.path.join(OUTPUT_DIR, "objects_assigned_to_polygons.csv"))

# =========================
# 17. COLORS
# =========================
valid_types = sorted(polygon_summary["best_type"].dropna().unique().tolist())

palette = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
    "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"
]

type_to_color = {t: palette[i % len(palette)] for i, t in enumerate(valid_types)}

# =========================
# 18. MAP
# =========================
center_lat = np.mean([lat for lat, lon in SADOVOE_POLYGON_LATLON])
center_lon = np.mean([lon for lat, lon in SADOVOE_POLYGON_LATLON])

m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")

# контур Садового
folium.Polygon(
    locations=SADOVOE_POLYGON_LATLON,
    color="black",
    weight=2,
    fill=False,
    popup="Approximate Sadovoe polygon"
).add_to(m)

# полигоны
for _, row in polygon_summary.iterrows():
    geom = polygon_geoms[row["polygon_id"]]

    if geom.is_empty:
        continue

    coords = []
    for x, y in np.array(geom.exterior.coords):
        lon, lat = xy_to_lonlat(x, y)
        coords.append([lat, lon])

    status = row["status"]

    if status == "ok":
        color = type_to_color.get(row["best_type"], "#999999")
        second_type_text = row["second_type"] if pd.notna(row["second_type"]) else "—"
        second_score_text = f"{row['second_score']:.3f}" if pd.notna(row["second_score"]) else "—"
        gap_text = f"{row['gap_1st_2nd']:.3f}" if pd.notna(row["gap_1st_2nd"]) else "—"
        overall_score_text = f"{row['overall_score']:.3f}" if pd.notna(row["overall_score"]) else "—"

        popup_html = (
            f"<b>Polygon:</b> {int(row['polygon_id'])}<br>"
            f"<b>Best type:</b> {row['best_type']}<br>"
            f"<b>Best score:</b> {row['best_score']:.3f}<br>"
            f"<b>Best type N:</b> {int(row['best_type_n'])}<br>"
            f"<b>Second type:</b> {second_type_text}<br>"
            f"<b>Second score:</b> {second_score_text}<br>"
            f"<b>Gap 1st-2nd:</b> {gap_text}<br>"
            f"<b>Total N:</b> {int(row['total_n']) if pd.notna(row['total_n']) else 0}<br>"
            f"<b>Overall score:</b> {overall_score_text}<br>"
            f"<b>Area:</b> {row['zone_area_m2']/1_000_000:.3f} km²<br><br>"
            f"<b>Top types:</b><br>{row['top_3_types'] if pd.notna(row['top_3_types']) else '—'}"
        )

        folium.Polygon(
            locations=coords,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.28,
            popup=folium.Popup(popup_html, max_width=360)
        ).add_to(m)

    else:
        popup_html = (
            f"<b>Polygon:</b> {int(row['polygon_id'])}<br>"
            f"<b>Status:</b> {status}<br>"
            f"<b>Total N:</b> {int(row['total_n']) if pd.notna(row['total_n']) else 0}<br>"
            f"<b>Area:</b> {row['zone_area_m2']/1_000_000:.3f} km²"
        )

        folium.Polygon(
            locations=coords,
            color="#bdbdbd",
            weight=1,
            fill=True,
            fill_color="#d9d9d9",
            fill_opacity=0.12,
            popup=folium.Popup(popup_html, max_width=320)
        ).add_to(m)

# центры
for _, row in zones_df.iterrows():
    folium.CircleMarker(
        location=[row["center_lat"], row["center_lon"]],
        radius=3,
        color="black",
        fill=True,
        fill_color="black",
        popup=f"Polygon center {int(row['polygon_id'])}"
    ).add_to(m)

# легенда
legend_html = """
<div style="
position: fixed;
bottom: 40px;
left: 40px;
z-index: 9999;
background-color: white;
padding: 12px;
border: 2px solid grey;
font-size: 14px;
max-height: 320px;
overflow-y: auto;
">
<b>Best type by polygon</b><br>
"""
for t in valid_types:
    legend_html += f'<i style="background:{type_to_color[t]};width:12px;height:12px;display:inline-block;margin-right:6px;"></i>{t}<br>'
legend_html += "</div>"

m.get_root().html.add_child(folium.Element(legend_html))

map_path = os.path.join(OUTPUT_DIR, "sadovoe_20_polygons_map.html")
m.save(map_path)

print("Saved map:")
print(map_path)

print("\nPreview:")
preview_cols = [
    "polygon_id", "best_type", "best_score",
    "second_type", "second_score",
    "gap_1st_2nd", "total_n", "status"
]
print(polygon_summary[preview_cols].sort_values("polygon_id").to_string(index=False))

print("\nDone.")