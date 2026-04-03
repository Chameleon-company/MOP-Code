const canvas = document.getElementById("bubbleCanvas");
const ctx = canvas.getContext("2d");
const scoreBoard = document.getElementById("scoreBoard");
const affirmationDiv = document.getElementById("affirmation");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let bubbles = [];
let particles = [];
let score = 0;
let affirmations = [];

fetch("/static/bubble_game/affirmations.json")
    .then(res => res.json())
    .then(data => affirmations = data);

const popSound = new Audio("/static/bubble_game/pop.mp3");
const ambient = new Audio("/static/bubble_game/ambient.mp3");
ambient.loop = true;
ambient.volume = 0.2;
ambient.play();

class Bubble {
    constructor() {
        this.reset();
    }

    reset() {
        this.radius = Math.random() * 30 + 20;
        this.x = Math.random() * canvas.width;
        this.y = canvas.height + this.radius;
        this.speed = Math.random() * 1.5 + 0.8 + score / 150;
        this.color = this.getGradientColor();
        this.popped = false;
    }

    getGradientColor() {
        const gradient = ctx.createRadialGradient(this.x, this.y, 10, this.x, this.y, this.radius);
        gradient.addColorStop(0, "#ffffffcc");
        gradient.addColorStop(1, "rgba(173, 216, 230, 0.7)");
        return gradient;
    }

    draw() {
        if (this.popped) return;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.fill();
        ctx.strokeStyle = "#b3e5fc";
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    update() {
        if (this.popped) return;
        this.y -= this.speed;
        if (this.y < -this.radius) this.reset();
    }

    burst(mx, my) {
        const dx = this.x - mx;
        const dy = this.y - my;
        return dx * dx + dy * dy < this.radius * this.radius;
    }
}

class Particle {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.radius = Math.random() * 2 + 1;
        this.speedX = (Math.random() - 0.5) * 5;
        this.speedY = (Math.random() - 0.5) * 5;
        this.life = 40;
    }

    draw() {
        ctx.globalAlpha = this.life / 40;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = "#81d4fa";
        ctx.fill();
        ctx.globalAlpha = 1.0;
    }

    update() {
        this.x += this.speedX;
        this.y += this.speedY;
        this.life--;
    }
}

function spawnBubbles() {
    if (bubbles.length < 30) {
        bubbles.push(new Bubble());
    }
}

function updateParticles() {
    particles = particles.filter(p => p.life > 0);
    for (let p of particles) {
        p.update();
        p.draw();
    }
}

function showAffirmation() {
    if (affirmations.length > 0) {
        const text = affirmations[Math.floor(Math.random() * affirmations.length)];
        affirmationDiv.innerText = text;
        affirmationDiv.style.opacity = 1;
        setTimeout(() => {
            affirmationDiv.style.opacity = 0;
        }, 3000);
    }
}

canvas.addEventListener("click", e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    for (let bubble of bubbles) {
        if (!bubble.popped && bubble.burst(mx, my)) {
            bubble.popped = true;
            popSound.currentTime = 0;
            popSound.play();
            score += 10;
            scoreBoard.innerText = `Score: ${score}`;
            for (let i = 0; i < 10; i++) {
                particles.push(new Particle(bubble.x, bubble.y));
            }
            showAffirmation();
        }
    }
});

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    spawnBubbles();
    for (let bubble of bubbles) {
        bubble.update();
        bubble.draw();
    }
    updateParticles();
    requestAnimationFrame(animate);
}

window.addEventListener("resize", () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});

animate();