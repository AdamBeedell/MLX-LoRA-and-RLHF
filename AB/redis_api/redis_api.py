import redis
import uuid
from fastapi import FastAPI
import redis
import uuid

app = FastAPI()

# Connect to Redis via Docker service name (important for Docker <-> Docker)
r = redis.Redis(host="redis", port=6379, decode_responses=True)

@app.get("/comparisons")
def get_all_comparisons():
    keys = r.keys("comparison:*")
    all_data = [r.hgetall(k) for k in keys]
    return all_data

@app.post("/comparison")
def store_comparison(preferred: str, other: str):
    key = f"comparison:{uuid.uuid4()}"
    r.hset(key, mapping={"preferred": preferred, "other": other})
    return {"key": key}

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

egpreferred = """
In July, London experiences warm and generally pleasant weather. The average temperature ranges from 60°F to 73°F (16°C to 23°C), with occasional spikes into the high 80s. The city gets an average of 6 hours of sunshine per day and only about 10 days of rainfall during the month. The evenings are typically mild, making it a great time to explore outdoor attractions and enjoy the city's parks and gardens. Overall, July is a popular month for tourists to visit London due to the favorable weather conditions.
"""

egother= """
London in July generally offers pleasant, warm summer weather, though it's famously changeable and unpredictable.

Here's a breakdown:

Temperature:

Daytime: Expect average high temperatures to range from 18°C to 25°C (64°F to 77°F).
Nighttime: Evenings are mild, typically falling to 12°C to 16°C (54°F to 61°F).
Heatwaves: While not the norm, London can experience brief heatwaves where temperatures soar into the high 20s or even low 30s °C (80s to low 90s °F). However, these are usually short-lived.
Sunshine & Daylight:

Long Days: July boasts very long daylight hours, with the sun rising early (around 5 AM) and setting late (around 9 PM), providing ample time for outdoor activities.
Bright Periods: There's usually plenty of sunshine, with many beautiful bright days.
Rainfall:

Showers are Common: This is where the "unpredictable" part comes in. Even in summer, London is prone to rain. You can expect sunny spells interspersed with occasional, brief showers.
Thunderstorms: Heavier downpours or even thunderstorms are possible, especially in the afternoon, but these typically pass quickly.
Not Usually All-Day Rain: While it can rain, it's rare for it to rain continuously all day in July. More often, it's a passing shower that clears up for more sunshine.
Humidity:

It can feel somewhat humid, especially before or during a shower, but generally, it's not excessively sticky.
In summary, expect:

Comfortably warm days perfect for exploring.
Mild evenings suitable for al fresco dining.
Long hours of daylight to maximize sightseeing.
A mix of sunshine and occasional, usually brief, rain showers.
What to pack:

Layers are key! A t-shirt for warm periods, a light cardigan or jumper for cooler evenings or cloudy spells.
Light waterproof jacket or a small umbrella - always a good idea, just in case.
Sunglasses and sunscreen for sunny periods.
Comfortable walking shoes.
"""

#store_comparison(preferred, other)


#r.hgetall("comparison:e5503728-d578-454d-88e2-bbcf51a90ceb")
#get_all_comparisons()