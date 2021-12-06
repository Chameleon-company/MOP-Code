from logic.parking_availability import (collect_parkingsensor,
                                        update_daily_parking)


def run(event, context):
    if event['name'] == "update_daily":
        return update_daily_parking()

    if event['name'] == "collect_parkingsensor":
        return collect_parkingsensor()

    if event['name'] == "test":
        return "Function test is working."

    raise Exception(f"Event name '{event['name']}' not recognized")
