import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO, solutions
import os

# create folder store car overspeed
cropped_images_dir = "cropped_vehicles"
os.makedirs(cropped_images_dir, exist_ok=True)

# send email
def send_email():
    sender_email = "kelvinbui0906115598@gmail.com"
    sender_password = "rrze hmvj vpud lbua"
    recipient_email = "kelvinbui0906115598@gmail.com"

    subject = "🚨 Urgent: Traffic Congestion Alert! 🚨"
    body = """
            <html>
            <body>
                <h2>🚨 <b>Traffic Alert: Potential Traffic Jam Detected on Flinders Street, Melbourne</b> 🚨</h2>
                <p>Dear Officer,</p>
                <p>Our advanced monitoring system has detected a <b>traffic congestion</b> situation that requires immediate attention. Here are the key details:</p>
                <p><b>🔴 Traffic Details:</b></p>
                <ul>
                    <li><b>Location:</b> Flinders Street, Melbourne</li>
                    <li><b>Number of vehicles:</b> More than 20</li>
                    <li><b>Slow-moving vehicles:</b> More than 10 vehicles traveling below 20 km/h</li>
                </ul>
                <p>This could indicate a possible <b>traffic jam</b> on the road. Immediate intervention is recommended to alleviate the situation and avoid further delays or accidents.</p>
                <p>Stay safe and thank you for your swift response to ensure smooth traffic flow!</p>
                <p>Sincerely,</p>
                <p>Traffic Monitoring System</p>
                <hr>
                <p><b>Note:</b> This is an automated alert from our system.</p>
            </body>
            </html>
            """

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to cut and save fast car image
def save_cropped_vehicle(image, box, vehicle_id, frame_id):
    """
    Save the cropped vehicle image to a folder.
    :param image: Original frame
    :param box: Bounding box (x1, y1, x2, y2)
    :param vehicle_id: Vehicle ID
    :param frame_id: Frame number
    """
    x1, y1, x2, y2 = map(int, box)
    # Make sure the bounding box is within the frame size
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    cropped_image = image[y1:y2, x1:x2]
    if cropped_image.size > 0:
        # Create file name with vehicle ID and frame number
        image_path = os.path.join(cropped_images_dir, f"vehicle_{vehicle_id}_frame_{frame_id}.jpg")
        cv2.imwrite(image_path, cropped_image)
        print(f"Saved cropped image for vehicle {vehicle_id} in frame {frame_id} at {image_path}")
    else:
        print(f"Invalid bounding box for vehicle {vehicle_id}. Image not saved.")

# Load YOLO model
model = YOLO("/Users/nguyenbui/Documents/upload_drive_crop_image/best.pt")
names = model.model.names

# Open video file
video_path = "/Users/nguyenbui/Documents/upload_drive_crop_image/highway_edit.mp4"
print(f"Video path: {video_path}")
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("police_vehicle_tracking_alert_2.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#threshold in city street video 0907(2).mp4
# counting_line_pts = [(20, 560), (1900, 560)]
# speed_line_pts = [(0, 120), (1900, 120)]

#threshold in highway video highway_edit.mp4
counting_line_pts = [(20, 560), (1900, 560)]
speed_line_pts = [(0, 320), (1900, 320)]

counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=counting_line_pts,
    names=names,
    draw_tracks=True,
    line_thickness=2,
)

speed_obj = solutions.SpeedEstimator(
    reg_pts=speed_line_pts,
    names=names,
    view_img=True,
)

vehicle_speeds = {}
sent_email = False
number_of_mail_send = 0
frame_id = 0  # Frame counter

# Main video processing loop
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    frame_id += 1  # Increment frame counter
    tracks = model.track(im0, persist=True, show=False)
    num_vehicles = len(tracks[0].boxes)
    print(f"Frame {frame_id}: Number of vehicles: {num_vehicles}")

    im0 = counter.start_counting(im0, tracks)
    im0 = speed_obj.estimate_speed(im0, tracks)

    slow_vehicles_count = 0
    for track in tracks[0].boxes:
        box = track.xyxy[0].tolist()
        vehicle_id = int(track.id)
        speed = speed_obj.dist_data.get(vehicle_id, 0)

        # Crop and save image for every frame if speed > 50 km/h
        if speed > 50:
            print(f"Vehicle ID {vehicle_id} exceeding speed in frame {frame_id}: {speed:.2f} km/h")
            save_cropped_vehicle(im0, box, vehicle_id, frame_id)

        # Count slow-moving vehicles
        if speed < 20:
            slow_vehicles_count += 1

    print(f"Slow vehicles: {slow_vehicles_count}, Total vehicles: {num_vehicles}")

    # Check for traffic congestion and send email
    if num_vehicles > 22 and slow_vehicles_count > 10:
        print("⚠️ Traffic Congestion Detected!")
        cv2.putText(im0, "Traffic Congestion", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        if not sent_email:
            send_email()
            sent_email = True
            number_of_mail_send += 1
    else:
        cv2.putText(im0, "Clear Traffic", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if number_of_mail_send == 0:
            sent_email = False

    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("\nFinal Vehicle Speeds:")
for vehicle_id, speed in vehicle_speeds.items():
    print(f"Vehicle ID: {vehicle_id}, Speed: {speed:.2f} km/h")

print("Vehicle counting and speed estimation completed. Video saved as 'police_vehicle_tracking_alert_2.avi'")
