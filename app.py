from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib
matplotlib.use("Agg")  #using a non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
import matplotlib.patches as mpatches

#creating Flask app
app = Flask(__name__)

#loading dataset
data = pd.read_csv("cleaned_data_with_splits.csv")

#filtering only for Male and Female in the Gender column
data = data[data["Gender"].isin(["Male", "Female"])]

print(data["Gender"].unique())
data.to_csv("gender_filtered_data.csv", index=False)

@app.route('/')
def home():
    #extracting unique values for dropdowns
    activity_types = data["ActivityType"].unique()
    response_types = data["Response"].unique()
    #passing the values to the template
    return render_template("index.html", activity_types=activity_types, response_types=response_types)

@app.route("/visualize", methods=["GET"])
def visualize():
    #getting user selections from dropdowns
    activity = request.args.get("activity")
    response = request.args.get("response")

    #filtering the data based on the user's selections
    filtered_data = data[(data["ActivityType"] == activity) & (data["Response"] == response)]

    #grouping by Year and Gender
    gender_trends = filtered_data.groupby(["Year", "Gender"])["Percentage"].mean().unstack()

    #creating the plot
    if not gender_trends.empty:
        plt.figure(figsize=(12, 7))  # Wider plot
        ax = gender_trends.plot(kind="bar", color={"Male": "cornflowerblue", "Female": "lightpink"}, alpha=0.85, edgecolor="black")

        #titles and labels
        plt.title(f"Visualization for {activity} ({response})", fontsize=7, pad=20, fontweight="bold")
        plt.xlabel("Year", fontsize=12, fontweight="medium")
        plt.ylabel("Percentage", fontsize=12, fontweight="medium")

        #gridlines
        plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)

        #fonts
        plt.xticks(fontsize=10, fontfamily="sans-serif")
        plt.yticks(fontsize=10, fontfamily="sans-serif")

        #legend
        plt.legend(title="Gender", fontsize=10, title_fontsize=12, loc="upper left", bbox_to_anchor=(1.02, 1))

        #background
        plt.gca().set_facecolor("#f7f7f7")
        plt.gca().spines["top"].set_visible(False)  # Hide top and right spines
        plt.gca().spines["right"].set_visible(False)

        #data Labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', label_type="edge", fontsize=10, padding=3)

        plt.tight_layout()  # Avoid overlap

        #saving the plot as a PNG image in-memory
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plt.close()

        #encoding the image to base64 for embedding in HTML
        plot_url = base64.b64encode(img.getvalue()).decode()

        #passing the image to the template
        return render_template("visualize.html", plot_url=plot_url, activity=activity, response=response)
    else:
        return render_template("no_data.html", activity=activity, response=response)


if __name__ == "__main__":
    app.run(debug=True)
