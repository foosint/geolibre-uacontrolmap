from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import geopandas as gpd
from shapely.geometry import (
    LineString,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

# ---------------------------------------------------------------------------
# KML namespaces
# ---------------------------------------------------------------------------

KML_NS = "http://www.opengis.net/kml/2.2"

NS = {
    "kml": KML_NS,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class KMLStyle:
    style_id: str

    # Line
    stroke: str | None = None
    stroke_opacity: float | None = None
    stroke_width: float | None = None

    # Polygon
    fill: str | None = None
    fill_opacity: float | None = None

    # Icon / point
    marker_color: str | None = None
    marker_scale: float | None = None
    marker_symbol: str | None = None
    icon_href: str | None = None

    # Label
    label_color: str | None = None
    label_scale: float | None = None

    # Original KML style id
    kml_style_id: str | None = None


@dataclass
class Feature:
    geometry: BaseGeometry
    name: str | None
    description: str | None

    folders: list[str] = field(default_factory=list)

    style_id: str | None = None
    style: KMLStyle | None = None

    properties: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def local_name(tag: str) -> str:
    """Return the local part of an XML tag."""

    if "}" in tag:
        return tag.rsplit("}", 1)[1]

    return tag


def text(element: ET.Element | None) -> str | None:
    """Return stripped element text."""

    if element is None or element.text is None:
        return None

    value = element.text.strip()

    return value if value else None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def strip_style_url(style_url: str | None) -> str | None:
    if not style_url:
        return None

    style_url = style_url.strip()

    if style_url.startswith("#"):
        return style_url[1:]

    # Handle URLs such as:
    # document.kml#style123
    if "#" in style_url:
        return style_url.rsplit("#", 1)[1]

    return style_url


def kml_color_to_rgba(
    value: str | None,
) -> tuple[str | None, float | None]:
    """
    Convert KML's AABBGGRR color notation to:

        (#RRGGBB, opacity)

    KML:
        ff0000ff = opaque red
        7f0000ff = ~50% red
    """

    if not value:
        return None, None

    value = value.strip().lstrip("#")

    if len(value) != 8:
        return None, None

    try:
        alpha = int(value[0:2], 16)
        blue = value[2:4]
        green = value[4:6]
        red = value[6:8]
    except ValueError:
        return None, None

    rgb = f"#{red}{green}{blue}"
    opacity = round(alpha / 255.0, 6)

    return rgb, opacity


def safe_filename(value: str) -> str:
    """
    Convert a folder path/name into a filesystem-safe filename.
    """

    value = value.strip()

    value = re.sub(
        r"[<>:\"/\\|?*]",
        "_",
        value,
    )

    value = re.sub(
        r"\s+",
        "_",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    )

    value = value.strip("._ ")

    if not value:
        return "layer"

    return value


# ---------------------------------------------------------------------------
# KML parsing
# ---------------------------------------------------------------------------


def find_first(
    element: ET.Element,
    name: str,
) -> ET.Element | None:
    return element.find(
        f".//kml:{name}",
        NS,
    )


def parse_coordinates(
    element: ET.Element,
) -> list[tuple[float, float]]:
    """
    Parse a KML <coordinates> element.

    KML coordinates are:

        longitude,latitude[,altitude]
    """

    value = text(element)

    if not value:
        return []

    coordinates: list[tuple[float, float]] = []

    for item in value.replace("\n", " ").split():

        parts = item.split(",")

        if len(parts) < 2:
            continue

        try:
            longitude = float(parts[0])
            latitude = float(parts[1])
        except ValueError:
            continue

        coordinates.append(
            (longitude, latitude)
        )

    return coordinates


def parse_point(
    element: ET.Element,
) -> Point | None:

    coordinates = find_first(
        element,
        "coordinates",
    )

    if coordinates is None:
        return None

    values = parse_coordinates(coordinates)

    if not values:
        return None

    return Point(values[0])


def parse_linestring(
    element: ET.Element,
) -> LineString | None:

    coordinates = find_first(
        element,
        "coordinates",
    )

    if coordinates is None:
        return None

    values = parse_coordinates(coordinates)

    if len(values) < 2:
        return None

    return LineString(values)


def parse_polygon(
    element: ET.Element,
) -> Polygon | None:

    outer = element.find(
        ".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates",
        NS,
    )

    if outer is None:
        return None

    exterior = parse_coordinates(outer)

    if len(exterior) < 3:
        return None

    holes: list[list[tuple[float, float]]] = []

    for inner in element.findall(
        ".//kml:innerBoundaryIs/kml:LinearRing/kml:coordinates",
        NS,
    ):
        values = parse_coordinates(inner)

        if len(values) >= 3:
            holes.append(values)

    return Polygon(
        exterior,
        holes,
    )


def parse_geometry(
    placemark: ET.Element,
) -> BaseGeometry | None:

    point = placemark.find(
        "kml:Point",
        NS,
    )

    if point is not None:
        return parse_point(point)

    line = placemark.find(
        "kml:LineString",
        NS,
    )

    if line is not None:
        return parse_linestring(line)

    polygon = placemark.find(
        "kml:Polygon",
        NS,
    )

    if polygon is not None:
        return parse_polygon(polygon)

    # MultiGeometry
    multi = placemark.find(
        "kml:MultiGeometry",
        NS,
    )

    if multi is not None:

        geometries: list[BaseGeometry] = []

        for child in multi:

            name = local_name(child.tag)

            if name == "Point":
                geometry = parse_point(child)

            elif name == "LineString":
                geometry = parse_linestring(child)

            elif name == "Polygon":
                geometry = parse_polygon(child)

            else:
                geometry = None

            if geometry is not None:
                geometries.append(geometry)

        if len(geometries) == 1:
            return geometries[0]

        if geometries:
            from shapely.geometry import GeometryCollection

            return GeometryCollection(geometries)

    return None


# ---------------------------------------------------------------------------
# ExtendedData
# ---------------------------------------------------------------------------


def parse_extended_data(
    placemark: ET.Element,
) -> dict[str, Any]:

    result: dict[str, Any] = {}

    extended = placemark.find(
        "kml:ExtendedData",
        NS,
    )

    if extended is None:
        return result

    # <Data name="foo">
    for data in extended.findall(
        "kml:Data",
        NS,
    ):

        name = data.attrib.get("name")

        if not name:
            continue

        value = text(
            data.find(
                "kml:value",
                NS,
            )
        )

        result[name] = value

    # Google My Maps commonly uses:
    #
    # <SchemaData schemaUrl="...">
    #     <SimpleData name="foo">bar</SimpleData>
    # </SchemaData>

    for simple in extended.findall(
        ".//kml:SimpleData",
        NS,
    ):

        name = simple.attrib.get("name")

        if not name:
            continue

        result[name] = text(simple)

    return result


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def parse_style(
    element: ET.Element,
) -> KMLStyle:

    style_id = element.attrib.get(
        "id",
        "",
    )

    style = KMLStyle(
        style_id=style_id,
        kml_style_id=style_id,
    )

    # LineStyle
    line = element.find(
        "kml:LineStyle",
        NS,
    )

    if line is not None:

        color = text(
            line.find(
                "kml:color",
                NS,
            )
        )

        style.stroke, style.stroke_opacity = (
            kml_color_to_rgba(color)
        )

        style.stroke_width = parse_float(
            text(
                line.find(
                    "kml:width",
                    NS,
                )
            )
        )

    # PolyStyle
    poly = element.find(
        "kml:PolyStyle",
        NS,
    )

    if poly is not None:

        color = text(
            poly.find(
                "kml:color",
                NS,
            )
        )

        style.fill, style.fill_opacity = (
            kml_color_to_rgba(color)
        )

    # IconStyle
    icon = element.find(
        "kml:IconStyle",
        NS,
    )

    if icon is not None:

        color = text(
            icon.find(
                "kml:color",
                NS,
            )
        )

        style.marker_color, _ = (
            kml_color_to_rgba(color)
        )

        style.marker_scale = parse_float(
            text(
                icon.find(
                    "kml:scale",
                    NS,
                )
            )
        )

        href = text(
            icon.find(
                "kml:Icon/kml:href",
                NS,
            )
        )

        style.icon_href = href

    # LabelStyle
    label = element.find(
        "kml:LabelStyle",
        NS,
    )

    if label is not None:

        color = text(
            label.find(
                "kml:color",
                NS,
            )
        )

        style.label_color, _ = (
            kml_color_to_rgba(color)
        )

        style.label_scale = parse_float(
            text(
                label.find(
                    "kml:scale",
                    NS,
                )
            )
        )

    return style


def parse_styles(
    root: ET.Element,
) -> dict[str, KMLStyle]:

    styles: dict[str, KMLStyle] = {}

    for element in root.findall(
        ".//kml:Style",
        NS,
    ):

        style_id = element.attrib.get("id")

        if not style_id:
            continue

        style = parse_style(element)

        styles[style_id] = style

    return styles


def parse_style_maps(
    root: ET.Element,
) -> dict[str, str]:

    """
    Resolve StyleMap normal/highlight references.

    We use the 'normal' style.
    """

    result: dict[str, str] = {}

    for style_map in root.findall(
        ".//kml:StyleMap",
        NS,
    ):

        style_id = style_map.attrib.get("id")

        if not style_id:
            continue

        normal_style: str | None = None

        for pair in style_map.findall(
            "kml:Pair",
            NS,
        ):

            key = text(
                pair.find(
                    "kml:key",
                    NS,
                )
            )

            if key != "normal":
                continue

            normal_style = strip_style_url(
                text(
                    pair.find(
                        "kml:styleUrl",
                        NS,
                    )
                )
            )

            break

        if normal_style:
            result[style_id] = normal_style

    return result


def resolve_style(
    style_id: str | None,
    styles: dict[str, KMLStyle],
    style_maps: dict[str, str],
) -> KMLStyle | None:

    if not style_id:
        return None

    style_id = strip_style_url(style_id)

    if not style_id:
        return None

    # Resolve StyleMap -> Style
    visited: set[str] = set()

    while style_id in style_maps:

        if style_id in visited:
            break

        visited.add(style_id)

        style_id = style_maps[style_id]

    style = styles.get(style_id)

    if style is None:
        return None

    return style


# ---------------------------------------------------------------------------
# KML traversal
# ---------------------------------------------------------------------------


def parse_placemark(
    placemark: ET.Element,
    folders: list[str],
    styles: dict[str, KMLStyle],
    style_maps: dict[str, str],
) -> Feature | None:

    geometry = parse_geometry(
        placemark
    )

    if geometry is None:
        return None

    name = text(
        placemark.find(
            "kml:name",
            NS,
        )
    )

    description = text(
        placemark.find(
            "kml:description",
            NS,
        )
    )

    style_url = strip_style_url(
        text(
            placemark.find(
                "kml:styleUrl",
                NS,
            )
        )
    )

    style = resolve_style(
        style_url,
        styles,
        style_maps,
    )

    properties = parse_extended_data(
        placemark
    )

    return Feature(
        geometry=geometry,
        name=name,
        description=description,
        folders=folders.copy(),
        style_id=style_url,
        style=style,
        properties=properties,
    )


def walk_container(
    element: ET.Element,
    folders: list[str],
    styles: dict[str, KMLStyle],
    style_maps: dict[str, str],
    output: list[Feature],
) -> None:

    for child in element:

        child_name = local_name(
            child.tag
        )

        if child_name == "Placemark":

            feature = parse_placemark(
                child,
                folders,
                styles,
                style_maps,
            )

            if feature is not None:
                output.append(feature)

        elif child_name == "Folder":

            folder_name = text(
                child.find(
                    "kml:name",
                    NS,
                )
            )

            new_folders = folders.copy()

            if folder_name:
                new_folders.append(
                    folder_name
                )

            walk_container(
                child,
                new_folders,
                styles,
                style_maps,
                output,
            )

        elif child_name == "Document":

            walk_container(
                child,
                folders,
                styles,
                style_maps,
                output,
            )


def parse_kml(
    kml_path: Path,
) -> list[Feature]:

    print(f"Reading KML: {kml_path}")

    tree = ET.parse(kml_path)

    root = tree.getroot()

    styles = parse_styles(root)
    style_maps = parse_style_maps(root)

    print(
        f"Found {len(styles)} styles "
        f"and {len(style_maps)} style maps."
    )

    features: list[Feature] = []

    walk_container(
        root,
        [],
        styles,
        style_maps,
        features,
    )

    print(
        f"Found {len(features)} features."
    )

    return features


# ---------------------------------------------------------------------------
# KMZ extraction
# ---------------------------------------------------------------------------


def extract_kmz(
    kmz_path: Path,
    destination: Path,
) -> Path:

    print(f"Extracting KMZ: {kmz_path}")

    if destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(kmz_path) as archive:

        archive.extractall(
            destination
        )

    # KMZ normally contains doc.kml.
    doc_kml = destination / "doc.kml"

    if doc_kml.exists():
        return doc_kml

    # Fall back to first KML file.
    kml_files = list(
        destination.rglob("*.kml")
    )

    if not kml_files:
        raise RuntimeError(
            "No KML file found inside KMZ."
        )

    return kml_files[0]


# ---------------------------------------------------------------------------
# Layer filtering
# ---------------------------------------------------------------------------


def folder_path(
    folders: list[str],
) -> str:

    return "/".join(folders)


def folder_is_selected(
    folders: list[str],
    include_folders: list[str],
    exclude_folders: list[str],
    include_descendants: bool,
) -> bool:

    path = folder_path(folders)

    # Root-level features
    if not path:
        path = "__root__"

    # Exclusions always win.
    for excluded in exclude_folders:

        excluded = excluded.strip(
            "/ "
        )

        if path == excluded:
            return False

        if path.startswith(
            excluded + "/"
        ):
            return False

    # Empty include list = everything.
    if not include_folders:
        return True

    for included in include_folders:

        included = included.strip(
            "/ "
        )

        if path == included:
            return True

        if (
            include_descendants
            and path.startswith(
                included + "/"
            )
        ):
            return True

    return False


# ---------------------------------------------------------------------------
# Feature -> GeoDataFrame
# ---------------------------------------------------------------------------


def feature_to_properties(
    feature: Feature,
) -> dict[str, Any]:

    result: dict[str, Any] = {}

    if feature.name is not None:
        result["name"] = feature.name

    if feature.description is not None:
        result["description"] = (
            feature.description
        )

    result["folder"] = folder_path(
        feature.folders
    )

    result["kml_style_id"] = (
        feature.style_id
    )

    style = feature.style

    if style is not None:

        result["stroke"] = style.stroke
        result["stroke_opacity"] = (
            style.stroke_opacity
        )
        result["stroke_width"] = (
            style.stroke_width
        )

        result["fill"] = style.fill
        result["fill_opacity"] = (
            style.fill_opacity
        )

        result["marker_color"] = (
            style.marker_color
        )
        result["marker_scale"] = (
            style.marker_scale
        )
        result["marker_symbol"] = (
            style.marker_symbol
        )

        result["icon_href"] = (
            style.icon_href
        )

        result["label_color"] = (
            style.label_color
        )
        result["label_scale"] = (
            style.label_scale
        )

    # Add ExtendedData.
    for key, value in feature.properties.items():

        # Avoid collisions with reserved
        # converter/style columns.
        if key in result:
            key = f"data_{key}"

        result[key] = value

    return result


def features_to_gdf(
    features: list[Feature],
) -> gpd.GeoDataFrame:

    rows: list[dict[str, Any]] = []

    for feature in features:

        row = feature_to_properties(
            feature
        )

        row["geometry"] = (
            feature.geometry
        )

        rows.append(row)

    if not rows:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Layer naming
# ---------------------------------------------------------------------------


def layer_name(
    folders: list[str],
) -> str:

    if not folders:
        return "root"

    return safe_filename(
        "__".join(folders)
    )


# ---------------------------------------------------------------------------
# GeoParquet
# ---------------------------------------------------------------------------


def write_layer(
    features: list[Feature],
    destination: Path,
) -> None:

    gdf = features_to_gdf(
        features
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Writing {len(gdf)} features:"
    )
    print(
        f"  {destination}"
    )

    gdf.to_parquet(
        destination,
        index=False,
        compression="zstd",
    )


# ---------------------------------------------------------------------------
# GeoLibre project
# ---------------------------------------------------------------------------


def relative_url(
    path: Path,
    base: Path,
) -> str:

    return path.relative_to(
        base
    ).as_posix()


def create_project(
    layers: list[dict[str, Any]],
    output_directory: Path,
    project_name: str,
) -> Path:

    """
    Generate a GeoLibre project.

    The generated project references the Parquet
    files using relative URLs so that the whole
    generated directory can be hosted by GitHub Pages.
    """

    project_path = (
        output_directory
        / f"{safe_filename(project_name)}.geolibre.json"
    )

    project_layers: list[dict[str, Any]] = []

    for layer in layers:

        project_layers.append(
            {
                "name": layer["name"],
                "type": "vector",
                "url": layer["url"],
            }
        )

    project = {
        "name": project_name,
        "layers": project_layers,
    }

    project_path.write_text(
        json.dumps(
            project,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return project_path


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def create_manifest(
    output_directory: Path,
    source_url: str,
    features: list[Feature],
    layers: list[dict[str, Any]],
) -> None:

    manifest = {
        "source_url": source_url,
        "feature_count": len(features),
        "layer_count": len(layers),
        "layers": layers,
    }

    path = (
        output_directory
        / "manifest.json"
    )

    path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(
    kmz_path: Path,
    config_path: Path,
) -> None:

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    output_directory = Path(
        config.get(
            "output_directory",
            "generated",
        )
    )

    project_name = config.get(
        "project_name",
        "GeoLibre project",
    )

    source_url = config.get(
        "source_url",
        "",
    )

    include_folders = config.get(
        "include_folders",
        [],
    )

    exclude_folders = config.get(
        "exclude_folders",
        [],
    )

    include_descendants = config.get(
        "include_descendants",
        True,
    )

    temporary_directory = (
        output_directory.parent
        / ".kmz_extract"
    )

    if temporary_directory.exists():
        shutil.rmtree(
            temporary_directory
        )

    # Keep generated output deterministic.
    layers_directory = (
        output_directory
        / "layers"
    )

    layers_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Extract KMZ.
    kml_path = extract_kmz(
        kmz_path,
        temporary_directory,
    )

    # Parse.
    all_features = parse_kml(
        kml_path
    )

    # Filter.
    selected_features = [
        feature
        for feature in all_features
        if folder_is_selected(
            feature.folders,
            include_folders,
            exclude_folders,
            include_descendants,
        )
    ]

    print(
        f"Selected {len(selected_features)} "
        f"of {len(all_features)} features."
    )

    # Group by folder.
    grouped: dict[str, list[Feature]] = {}

    for feature in selected_features:

        path = folder_path(
            feature.folders
        )

        grouped.setdefault(
            path,
            [],
        ).append(feature)

    # Remove old Parquet files.
    for old_file in layers_directory.glob(
        "*.parquet"
    ):
        old_file.unlink()

    layers: list[dict[str, Any]] = []

    for folder, folder_features in sorted(
        grouped.items()
    ):

        filename = (
            layer_name(
                folder_features[0].folders
            )
            + ".parquet"
        )

        parquet_path = (
            layers_directory
            / filename
        )

        write_layer(
            folder_features,
            parquet_path,
        )

        layers.append(
            {
                "name": folder
                if folder
                else "root",
                "folder": folder,
                "file": (
                    f"layers/{filename}"
                ),
                "url": (
                    f"layers/{filename}"
                ),
                "feature_count": len(
                    folder_features
                ),
            }
        )

    # Create GeoLibre project.
    project_path = create_project(
        layers,
        output_directory,
        project_name,
    )

    # Create manifest.
    create_manifest(
        output_directory,
        source_url,
        all_features,
        layers,
    )

    # Clean temporary extraction.
    if temporary_directory.exists():
        shutil.rmtree(
            temporary_directory
        )

    print()
    print("Conversion complete.")
    print(
        f"Project: {project_path}"
    )
    print(
        f"Layers:  {len(layers)}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Convert a Google My Maps KMZ "
            "to GeoParquet + GeoLibre project."
        )
    )

    parser.add_argument(
        "kmz",
        type=Path,
        help="Input KMZ file.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config.json"
        ),
        help="Configuration JSON.",
    )

    args = parser.parse_args()

    if not args.kmz.exists():
        parser.error(
            f"KMZ does not exist: {args.kmz}"
        )

    if not args.config.exists():
        parser.error(
            f"Config does not exist: "
            f"{args.config}"
        )

    convert(
        args.kmz,
        args.config,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())