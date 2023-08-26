import pycountry
from geopy.geocoders import Nominatim
import datetime
import reverse_geocode


def convert_utc_timestamp(utc_timestamp):
    # Convert UTC timestamp to datetime object
    utc_datetime = datetime.datetime.utcfromtimestamp(utc_timestamp)

    # Convert UTC datetime to local datetime
    local_datetime = utc_datetime.replace(tzinfo=datetime.timezone.utc).astimezone()

    # Format the local datetime as desired (year-month-day hour:minute:second)
    formatted_datetime = local_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")

    season, day, month, year, hour = fmow_timestamp2symdh(formatted_datetime)

    return season, day, month, year


def convert_yfcc_timestamp(timestamp):
    """

    :param timestamp: '2013-11-24 23:08:45.0'
    :return:
    """
    # Convert UTC timestamp to datetime object
    dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")

    # Define seasons by month
    seasons = {
        "Winter": (12, 1, 2),
        "Spring": (3, 4, 5),
        "Summer": (6, 7, 8),
        "Fall": (9, 10, 11)
    }

    # Determine the season
    season = next((s for s, months in seasons.items() if dt.month in months), None)

    # Get day, month name, year, and hour
    day = str(dt.day)
    month = dt.strftime('%B')
    year = str(dt.year)
    hour = str(dt.hour)

    return season, day, month, year, hour


def countrycode2countryname(country_code):
    """

    :param country_code: TUR
    :return: Turkey
    """
    try:
        country = pycountry.countries.get(alpha_3=country_code)
        return country.name
    except AttributeError:
        return "Unknown country code"


def latlon_to_city_country(latitude, longitude, country_name=None):
    # geolocator = Nominatim(user_agent="geotag")
    # location = geolocator.reverse((longitude, latitude), exactly_one=True)
    # raw_addr = location.raw['address']
    # province = raw_addr.get('province', None)
    # city = raw_addr.get('city', None)
    # if not city:  # sometimes the city might be stored as 'town' or 'village'
    #     city = raw_addr.get('town', None)
    # if not city:
    #     city = raw_addr.get('village', None)
    # out = country_name
    # if province:
    #     out = province + ", " + out
    # if city:
    #     out = city + ", " + out
    # fmow
    if country_name is not None:
        out = country_name
        coordinates = [(latitude, longitude)]
        city = reverse_geocode.search(coordinates)[0]["city"]
        if city:
            out = city + ", " + out

        # print(latitude, longitude, out)
        return out
    # yfcc
    else:
        coordinates = [(latitude, longitude)]
        country = reverse_geocode.search(coordinates)[0]["country"]
        out = country
        city = reverse_geocode.search(coordinates)[0]["city"]
        if city:
            out = city + ", " + out
        return out


def parse_coord(coord_text):
    """

    :param coord_text: "POLYGON ((73.9045104783130000 18.5879324392585232, 73.9333264511109292 18.5879324392585232,
    73.9333264511109292 18.5728217447537176, 73.9045104783130000 18.5728217447537176, 73.9045104783130000 18.5879324392585232))"

    :return:
    """
    # latitude = 18.5879324392585232
    # longitude = 73.9045104783130000

    longitude, latitude = coord_text.split(",")[0].split("((")[1].split(" ")
    return float(latitude), float(longitude)


def fmow_timestamp2symdh(timestamp):
    """

    :param timestamp: '2002-02-07T08:43:59Z'
    :return: (season, day, month, year, hour)
    """

    # Convert the string to a datetime object
    try:
        dt = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except:
        dt = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
        # print("not standard timestamp {}".format(timestamp))

    # Define seasons by month
    seasons = {
        "Winter": (12, 1, 2),
        "Spring": (3, 4, 5),
        "Summer": (6, 7, 8),
        "Fall": (9, 10, 11)
    }

    # Determine the season
    season = next((s for s, months in seasons.items() if dt.month in months), None)

    # Get day, month name, year, and hour
    day = str(dt.day)
    month = dt.strftime('%B')
    year = str(dt.year)
    hour = str(dt.hour)

    return season, day, month, year, hour


def fmow_bbox_location(image_width, image_height, bbox):
    """

    :param image_width:
    :param image_height:
    :param bbox: (up left x, up left y, w, h)
    :return:
    """
    # Define the 3x3 grid dimensions
    grid_width = image_width / 3
    grid_height = image_height / 3

    # Extract bbox details
    x, y, w, h = bbox

    # Define the boundaries for each of the 9 regions
    regions = {
        "Top-left":      (grid_width * 0, grid_height * 0, grid_width, grid_height),
        "Top-center":    (grid_width * 1, grid_height * 0, grid_width, grid_height),
        "Top-right":     (grid_width * 2, grid_height * 0, grid_width, grid_height),
        "Center-left":   (grid_width * 0, grid_height * 1, grid_width, grid_height),
        "Center":        (grid_width * 1, grid_height * 1, grid_width, grid_height),
        "Center-right":  (grid_width * 2, grid_height * 1, grid_width, grid_height),
        "Bottom-left":   (grid_width * 0, grid_height * 2, grid_width, grid_height),
        "Bottom-center": (grid_width * 1, grid_height * 2, grid_width, grid_height),
        "Bottom-right":  (grid_width * 2, grid_height * 2, grid_width, grid_height)
    }

    def intersection_area(target_bbox, region_bbox):
        """

        :param target_bbox: x1, y1, w1, h1
        :param region_bbox: x2, y2, w2, h2
        :return:
        """

        x1, y1, w1, h1 = target_bbox
        x2, y2, w2, h2 = region_bbox

        # Calculate the overlap boundaries
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)
        intersection_area = max(0, xB - xA) * max(0, yB - yA)

        return intersection_area

    # Calculate intersection area for each region
    overlaps = {
        region: intersection_area([x, y, w, h], [rx, ry, rw, rh])
        for region, (rx, ry, rw, rh) in regions.items()
    }

    # Return the region with the maximum overlap
    first_return = max(overlaps, key=overlaps.get)
    del overlaps[max(overlaps, key=overlaps.get)]
    second_return = max(overlaps, key=overlaps.get)
    return "{} and {} blocks".format(first_return, second_return)


def parse_fmow_df(gt_json, caption_template):
    # "from high above {country} during {season},
    # the image captured {class_label} residing at the {relative_location}.
    # its resolution is denoted with a ground sample distance of {ground_sample_distance}m.
    # the geo-tag is referenced to utm zone {utm_zone} with a date stamp of {timestamp}.
    # clouds made up {cloud_cover_rate}% of the sky, with the scan direction set {scan_direction}.
    # orientation angles stood at target azimuth: {target_azimuth}° and off-nadir: {off_nadir}°.",
    country_code = gt_json["country_code"]
    country_name = countrycode2countryname(country_code)
    latitude, longitude = parse_coord(gt_json["bounding_boxes"][0]["raw_location"])
    city_country_name = latlon_to_city_country(latitude, longitude, country_name)
    timestamp = gt_json["timestamp"]
    season, day, month, year, hour = fmow_timestamp2symdh(timestamp)
    bbox = gt_json["bounding_boxes"]
    if len(bbox) > 2:
        print("More than one bbox, take the first")
        print(gt_json)
        print()
    class_label = bbox[1]["category"]
    bbox_0 = bbox[1]["box"]
    image_width = gt_json["img_width"]
    image_height = gt_json["img_height"]
    relative_location = fmow_bbox_location(image_width, image_height, bbox_0)

    ground_sample_distance = gt_json["multi_resolution_dbl"]
    utm_zone = gt_json["utm"]
    cloud_cover_rate = gt_json["cloud_cover"]
    scan_direction = gt_json["scan_direction"]
    target_azimuth = gt_json["target_azimuth_dbl"]
    off_nadir = gt_json["off_nadir_angle_dbl"]

    caption = caption_template \
        .replace("{country}", city_country_name) \
        .replace("{season}", season) \
        .replace("{class_label}", class_label.lower()) \
        .replace("{relative_location}", relative_location.lower()) \
        .replace("{ground_sample_distance}", "%.2f" % ground_sample_distance) \
        .replace("{utm_zone}", utm_zone) \
        .replace("{timestamp}", "{} o'clock, {} {}, {}".format(hour, month, day, year)) \
        .replace("{cloud_cover_rate}", str(cloud_cover_rate)) \
        .replace("{scan_direction}", scan_direction.lower()) \
        .replace("{target_azimuth}", "%.2f" % target_azimuth) \
        .replace("{off_nadir}", "%.2f" % off_nadir)

    return caption, country_name, month, (latitude, longitude), utm_zone


def ben_timestamp2symdh(timestamp):
    """

    :param timestamp: '2017-10-02 11:21:12'
    :return: (season, day, month, year, hour)
    """

    # Convert the string to a datetime object
    dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    # Define seasons by month
    seasons = {
        "Winter": (12, 1, 2),
        "Spring": (3, 4, 5),
        "Summer": (6, 7, 8),
        "Fall": (9, 10, 11)
    }

    # Determine the season
    season = next((s for s, months in seasons.items() if dt.month in months), None)

    # Get day, month name, year, and hour
    day = str(dt.day)
    month = dt.strftime('%B')
    year = str(dt.year)
    hour = str(dt.hour)

    return season, day, month, year, hour


def ben_projection2utmzone(projection):
    """

    :param projection: PROJCS["WGS 84 / UTM zone 29N",
    GEOGCS["WGS 84",DATUM["WGS_1984",
    SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],
    AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],
    AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]],
    PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",-9],PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],PARAMETER["false_northing",0],
    UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32629"]]

    :return: utm zone (29N)
    """

    utm_zone = projection.split(",")[0].split("UTM zone ")[1][:-1]
    return utm_zone


def parse_ben_df(ben_info, caption_template):
    """

    :param ben_info:
    :param caption_template:
    :return:
    """
    ben_info = ben_info.to_numpy()[0]
    img_name, class_label, _, _, _, _, _, _, projection, acquisition_time = ben_info
    class_label = eval(class_label)
    class_label = ", ".join(class_label).lower().replace("_", "-")
    season, day, month, year, hour = ben_timestamp2symdh(acquisition_time)
    utm_zone = ben_projection2utmzone(projection)

    # "this satellite image, captured during the {season}, showcases the '{class_labels}' from utm zone {utm_zone} and is timestamped {timestamp}."

    caption = caption_template \
        .replace("{season}", season) \
        .replace("{class_labels}", class_label) \
        .replace("{utm_zone}", utm_zone) \
        .replace("{timestamp}", "{} o'clock, {} {}, {}".format(hour, month, day, year))
    return caption, month, utm_zone


def draw_bbox(image_path, bbox, color=(0, 0, 255), thickness=20, show_nine_block=True):
    """
    Draw a bounding box on the given image.

    Parameters:
    - image_path: path to the input image.
    - bbox: bounding box in format (x, y, w, h).
    - color: color of the bounding box in BGR format. Default is red.
    - thickness: line thickness. Default is 2.

    Returns:
    - The image with the bounding box.
    """
    import cv2
    # Read the image
    img = cv2.imread(image_path)

    # Draw the bounding box
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    if show_nine_block:
        image_height, image_width, c = img.shape
        grid_width = image_width / 3
        grid_height = image_height / 3
        regions = {
            "Top-left": (grid_width * 0, grid_height * 0, grid_width, grid_height),
            "Top-center": (grid_width * 1, grid_height * 0, grid_width, grid_height),
            "Top-right": (grid_width * 2, grid_height * 0, grid_width, grid_height),
            "Center-left": (grid_width * 0, grid_height * 1, grid_width, grid_height),
            "Center": (grid_width * 1, grid_height * 1, grid_width, grid_height),
            "Center-right": (grid_width * 2, grid_height * 1, grid_width, grid_height),
            "Bottom-left": (grid_width * 0, grid_height * 2, grid_width, grid_height),
            "Bottom-center": (grid_width * 1, grid_height * 2, grid_width, grid_height),
            "Bottom-right": (grid_width * 2, grid_height * 2, grid_width, grid_height)
        }
        for key in regions:
            coord = regions[key]
            cv2.rectangle(img, (int(coord[0]), int(coord[1])), (int(coord[0]) + int(coord[2]), int(coord[1]) + int(coord[3])), (0, 0, 0), thickness-10)

    # Display the image
    cv2.imshow('Image with BBox', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img


if __name__ == "__main__":
    # Example usage:
    # image_path = "/Users/zilun/Desktop/airport_0_0_msrgb.jpg"
    # bbox = [174, 363, 1429, 661] # Example bounding box (x, y, w, h)
    # draw_bbox(image_path, bbox, show_nine_block=True)

    latitude = 32.669976480324785
    longitude = 39.959161146012924

    # city_name = latlon_to_city_country(latitude, longitude, "TUK")
    # print(f"The city at coordinates {latitude}, {longitude} is {city_name}.")

    from transformers import GPT2Tokenizer
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    # Your text
    text = "from the vastness of space, the satellite captured a glimpse of North Saanich, Canada during its Summer. it specifically highlighted airport_hangar at the center and top-left blocks. this image's resolution is impressive, with a ground sample distance of 2.33m. it's catalogued under utm zone 10U, with the exact moment captured at 19 o'clock, July 29, 2016. conditions during capture were a 0% cloud cover and the scan was in the forward direction. guiding angles for this shot were target azimuth: 87.77° and off-nadir: 27.81°."
    # Tokenize
    tokens = tokenizer.tokenize(text)
    # Print number of tokens
    print(len(tokens))

    # coordinates = [(longitude, latitude)]
    # out = reverse_geocode.search(coordinates)
    # print(out)







