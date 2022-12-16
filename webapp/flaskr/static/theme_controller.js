const colors = [
  {
    id: "--color-one",
    light: "var(--go-green)",
    dark: "var(--medium-spring-green)",
  },
  {
    id: "--color-two",
    light: "var(--persian-green)",
    dark: "var(--medium-aquamarine)",
  },
  {
    id: "--color-three",
    light: "var(--viridian-green)",
    dark: "var(--turquoise)",
  },
  {
    id: "--color-four",
    light: "var(--blue-sapphire)",
    dark: "var(--robin-egg-blue)",
  },
  {
    id: "--color-five",
    light: "var(--smoky-black)",
    dark: "var(--pacific-blue)",
  },
  {
    id: "--color-six",
    light: "var(--missing-color)",
    dark: "var(--cyan-process)",
  },
  {
    id: "--default-text-color",
    light: "#ffffff",
    dark: "#000000",
  },
  {
    id: "--default-panel-color",
    light: "#3A3B3C",
    dark: "white",
},
{
    id: "--default-link-color",
    light: "cyan",
    dark: "#0000EE",
}
];

let count = 0;

window.addEventListener("DOMContentLoaded", (event) => {
  //This is needed to ensure the button is loaded prior to trying to access it
  console.log("DOM fully loaded and parsed");

  document.getElementById("dt-btn").addEventListener("click", () => {
    colors.forEach((color) => {
      document.documentElement.style.setProperty(color.id, count %2 === 0? color.light : color.dark);
      var img = document.getElementById('dt-btn').src.includes('moon_ico.png')
        if (img == true) {
            document.getElementById('dt-btn').src = "/static/sun_ico.png";
        } else {
           document.getElementById('dt-btn').src = "/static/moon_ico.png";
        }
    });
    count++;
  });
});