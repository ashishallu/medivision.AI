/*
particles.js
Ambient "neural HUD" backdrop: faint drifting nodes with connecting
lines when close to each other, rendered on a fixed full-screen canvas
behind the Streamlit app. Deliberately subtle and slow so it reads as
atmosphere, not decoration that competes with the content.

Injected into the Streamlit app via components.html().
*/

(function () {
    const canvas = document.getElementById('mv-particle-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let width, height;
    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    const NODE_COUNT = 46;
    const LINK_DIST = 130;
    const nodes = Array.from({ length: NODE_COUNT }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.18,
        vy: (Math.random() - 0.5) * 0.18,
    }));

    function step() {
        ctx.clearRect(0, 0, width, height);

        for (const n of nodes) {
            n.x += n.vx;
            n.y += n.vy;
            if (n.x < 0 || n.x > width) n.vx *= -1;
            if (n.y < 0 || n.y > height) n.vy *= -1;
        }

        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < LINK_DIST) {
                    const alpha = (1 - dist / LINK_DIST) * 0.12;
                    ctx.strokeStyle = `rgba(34, 211, 238, ${alpha})`;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.stroke();
                }
            }
        }

        for (const n of nodes) {
            ctx.beginPath();
            ctx.arc(n.x, n.y, 1.4, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(34, 211, 238, 0.45)';
            ctx.fill();
        }

        requestAnimationFrame(step);
    }
    step();
})();
