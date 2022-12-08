window.addEventListener('DOMContentLoaded', (event) => { //This is needed to ensure the button is loaded prior to trying to access it
  console.log('DOM fully loaded and parsed');

  //Set Dark Theme
  document.getElementById('dt-btn').addEventListener('click', () => {
    document.documentElement.style.setProperty('--color-one', 'var(--go-green)');
    document.documentElement.style.setProperty('--color-two', 'var(--persian-green)');
    document.documentElement.style.setProperty('--color-three', 'var(--viridian-green)');
    document.documentElement.style.setProperty('--color-four', 'var(--blue-sapphire)');
    document.documentElement.style.setProperty('--color-five', 'var(--smoky-black)');
    document.documentElement.style.setProperty('--color-six', 'var(--missing-color)');

    console.log('dark theme executed...');
  });

  //Set Light Theme
  document.getElementById('lt-btn').addEventListener('click', () => {
    document.documentElement.style.setProperty('--color-one', 'var(--medium-spring-green)');
    document.documentElement.style.setProperty('--color-two', 'var(--medium-aquamarine)');
    document.documentElement.style.setProperty('--color-three', 'var(--turquoise)');
    document.documentElement.style.setProperty('--color-four', 'var(--robin-egg-blue)');
    document.documentElement.style.setProperty('--color-five', 'var(--pacific-blue)');
    document.documentElement.style.setProperty('--color-six', 'var(--cyan-process)');
    console.log('light theme executed...');
  });
});



