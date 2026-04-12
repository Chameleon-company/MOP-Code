import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO, solutions

# Function to send email
def send_email():
    sender_email = "kelvinbui0906115598@gmail.com"  # implement your gmail
    sender_password = "rrze hmvj vpud lbua"  # implement your password in mail app
    recipient_email = "kelvinbui0906115598@gmail.com"  # implement receiver mail

    # Setup email subject and body
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

    # Create email
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "html"))

    # Send email via SMTP server (Gmail example)
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

# Load model
model = YOLO(r"C:\car-models.v2i.yolov8\runs\detect\train\weights\best.pt")
names = model.model.names

# Open video file
cap = cv2.VideoCapture(r"C:\car-models.v2i.yolov8\0907(2).mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Store the output after finish processing
video_writer = cv2.VideoWriter("police_vehicle_tracking_alert.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps,
                               (w, h))

# Line points for counting and speed estimatio  n 360
counting_line_pts = [(20, 560), (1900, 560)]
speed_line_pts = [(0, 120), (1900, 120)]

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=counting_line_pts,
    names=names,
    draw_tracks=True,
    line_thickness=2,
)

# Init Speed Estimator
speed_obj = solutions.SpeedEstimator(
    reg_pts=speed_line_pts,
    names=names,
    view_img=True,
)

# Dictionary to store vehicle speeds
vehicle_speeds = {}

# Email sent flag
sent_email = False
number_of_mail_send = 0

# Processing video frames
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform tracking
    tracks = model.track(im0, persist=True, show=False)

    # Count vehicles in the current frame
    num_vehicles = len(tracks[0].boxes)  # Assuming tracks[0] contains all detected objects
    print(f"Number of vehicles in frame: {num_vehicles}")

    # Count vehicles
    im0 = counter.start_counting(im0, tracks)

    # Estimate speed and store in dictionary
    im0 = speed_obj.estimate_speed(im0, tracks)

    # Retrieve speed data after estimation
    slow_vehicles_count = 0
    for vehicle_id, speed in speed_obj.dist_data.items():
        # Store the speed in the dictionary
        vehicle_speeds[vehicle_id] = speed
        # Print speed for each vehicle
        print(f"Vehicle ID: {vehicle_id}, Speed: {speed:.2f} km/h")
        # Check if speed less than 20km/h
        if speed < 20:
            slow_vehicles_count += 1

    # Print traffic jam condition
    print(f"Slow vehicles: {slow_vehicles_count}, Total vehicles: {num_vehicles}")
    # Check if there is a traffic jam and send email if detected
    if num_vehicles > 22 and slow_vehicles_count > 10:
        print("⚠️Traffic Congestion: Number of vehicles is more than 10 and their speed is less than 20 km/h")
        # Display "Traffic Congestion" on the top left corner of the video
        cv2.putText(im0, "Traffic Congestion", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Send email if it hasn't been sent already
        if not sent_email:
            #send mail if detect traffic congestion
            send_email()
            # Set flag to prevent multiple emails for the same jam
            sent_email = True
            number_of_mail_send = number_of_mail_send +1

    else:
        # Display "Clear Traffic" on the top left corner of the video
        cv2.putText(im0, "Clear Traffic", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Reset email flag if the traffic jam is cleared
        if number_of_mail_send == 0:
            sent_email = False

    # Write the processed frame to the output video
    video_writer.write(im0)

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Print all stored vehicle speeds
print("\nFinal Vehicle Speeds:")
for vehicle_id, speed in vehicle_speeds.items():
    print(f"Vehicle ID: {vehicle_id}, Speed: {speed:.2f} km/h")

print("Vehicle counting and speed estimation completed. Video saved as 'police_vehicle_tracking_alert.avi'")
