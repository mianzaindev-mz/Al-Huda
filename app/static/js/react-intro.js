/**
 * Al-Huda — Premium React Intro Animation
 * Islamic-themed launch screen with starfield, geometric patterns,
 * light rays, and smooth phase transitions.
 */
const { useState, useEffect, useRef, useMemo, createElement: h } = React;

// ─────────────────────────────────────────────
// Starfield — twinkling stars in the background
// ─────────────────────────────────────────────
function Starfield({ count }) {
    const stars = useMemo(() =>
        Array.from({ length: count }, (_, i) => ({
            x: Math.random() * 100,
            y: Math.random() * 100,
            size: 1 + Math.random() * 2.5,
            delay: Math.random() * 4,
            duration: 2 + Math.random() * 3,
            brightness: 0.3 + Math.random() * 0.7,
        })), [count]);

    return h('div', { style: { position: 'absolute', inset: 0, overflow: 'hidden' } },
        stars.map((s, i) => h('div', {
            key: i,
            style: {
                position: 'absolute',
                left: s.x + '%',
                top: s.y + '%',
                width: s.size,
                height: s.size,
                borderRadius: '50%',
                background: `rgba(255, 255, 255, ${s.brightness * 0.8})`,
                boxShadow: `0 0 ${s.size * 3}px rgba(16, 185, 129, ${s.brightness * 0.4})`,
                animation: `starTwinkle ${s.duration}s ease-in-out ${s.delay}s infinite`,
            },
        }))
    );
}

// ─────────────────────────────────────────────────
// Islamic Geometric Pattern — rotating octagon ring
// ─────────────────────────────────────────────────
function GeometricRing({ size, duration, reverse, opacity, borderStyle }) {
    // Create an octagonal shape using clip-path
    const style = {
        position: 'absolute',
        width: size,
        height: size,
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        border: borderStyle || '1.5px solid rgba(16, 185, 129, 0.2)',
        borderRadius: '50%',
        animation: `orbit ${duration}s linear infinite ${reverse ? 'reverse' : ''}`,
        opacity: opacity || 0.3,
    };
    return h('div', { style });
}

// ────────────────────────────────────────
// Light Ray — radiating beams from center
// ────────────────────────────────────────
function LightRays({ visible }) {
    const rays = useMemo(() =>
        Array.from({ length: 12 }, (_, i) => ({
            angle: (360 / 12) * i,
            width: 1 + (i % 3) * 0.5,
            length: 120 + (i % 4) * 40,
            delay: i * 0.08,
            opacity: 0.08 + (i % 3) * 0.04,
        })), []);

    return h('div', {
        style: {
            position: 'absolute',
            top: '50%',
            left: '50%',
            width: 0,
            height: 0,
            opacity: visible ? 1 : 0,
            transition: 'opacity 1.2s ease-out 0.5s',
        }
    },
        rays.map((r, i) => h('div', {
            key: i,
            style: {
                position: 'absolute',
                width: r.width,
                height: r.length,
                background: `linear-gradient(to bottom, rgba(16, 185, 129, ${r.opacity}), transparent)`,
                transformOrigin: '50% 0%',
                transform: `rotate(${r.angle}deg)`,
                opacity: visible ? 1 : 0,
                transition: `opacity 0.6s ease ${r.delay}s, height 0.8s ease ${r.delay}s`,
            },
        }))
    );
}

// ────────────────────────────────────────────────
// Floating Motes — warm golden particles drifting
// ────────────────────────────────────────────────
function FloatingMotes({ count, visible }) {
    const motes = useMemo(() =>
        Array.from({ length: count }, (_, i) => ({
            x: 10 + Math.random() * 80,
            delay: Math.random() * 5,
            size: 2 + Math.random() * 4,
            duration: 5 + Math.random() * 7,
            drift: -30 + Math.random() * 60,
            color: i % 3 === 0
                ? 'rgba(217, 171, 95, 0.5)'   // Gold
                : i % 3 === 1
                    ? 'rgba(16, 185, 129, 0.4)'  // Emerald
                    : 'rgba(52, 211, 153, 0.35)', // Teal
        })), [count]);

    return h('div', {
        style: {
            position: 'absolute', inset: 0, overflow: 'hidden',
            opacity: visible ? 1 : 0,
            transition: 'opacity 1s ease 0.3s',
        }
    },
        motes.map((m, i) => h('div', {
            key: i,
            style: {
                position: 'absolute',
                left: m.x + '%',
                bottom: '-5%',
                width: m.size,
                height: m.size,
                borderRadius: '50%',
                background: m.color,
                boxShadow: `0 0 ${m.size * 3}px ${m.color}`,
                animation: `moteRise ${m.duration}s ease-out ${m.delay}s infinite`,
            },
        }))
    );
}

// ───────────────────────────────────────────────
// Pulsing Halo — concentric expanding rings
// ───────────────────────────────────────────────
function PulsingHalos({ visible }) {
    const halos = [
        { delay: 0, color: 'rgba(5, 150, 105, 0.25)' },
        { delay: 1.2, color: 'rgba(16, 185, 129, 0.18)' },
        { delay: 2.4, color: 'rgba(52, 211, 153, 0.12)' },
    ];

    return h('div', { style: { position: 'absolute', top: '50%', left: '50%' } },
        halos.map((halo, i) => h('div', {
            key: i,
            style: {
                position: 'absolute',
                top: 0,
                left: 0,
                transform: 'translate(-50%, -50%)',
                borderRadius: '50%',
                border: `1.5px solid ${halo.color}`,
                animation: visible ? `haloExpand 3.6s ease-out ${halo.delay}s infinite` : 'none',
                opacity: visible ? 1 : 0,
            },
        }))
    );
}

// ═══════════════════════════════════════════════
// MAIN INTRO COMPONENT
// ═══════════════════════════════════════════════
function IntroAnimation({ onComplete }) {
    const [phase, setPhase] = useState('entering');  // entering → reveal → radiate → exiting → done
    const [fontsReady, setFontsReady] = useState(false);

    // Font loading
    useEffect(() => {
        const load = async () => {
            try {
                await document.fonts.ready;
                setFontsReady(true);
                document.body.classList.add('fonts-loaded');
            } catch { setFontsReady(true); document.body.classList.add('fonts-loaded'); }
        };
        load();
        setTimeout(() => { setFontsReady(true); document.body.classList.add('fonts-loaded'); }, 500);
    }, []);

    // Phase timeline
    useEffect(() => {
        document.body.style.opacity = '1';
        const t1 = setTimeout(() => setPhase('reveal'), 400);
        const t2 = setTimeout(() => setPhase('radiate'), 1600);
        const t3 = setTimeout(() => setPhase('exiting'), 4000);
        const t4 = setTimeout(() => { setPhase('done'); if (onComplete) onComplete(); }, 4800);
        return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); clearTimeout(t4); };
    }, []);

    if (phase === 'done') return null;

    const isRevealed = phase === 'reveal' || phase === 'radiate' || phase === 'exiting';
    const isRadiating = phase === 'radiate' || phase === 'exiting';
    const isExiting = phase === 'exiting';

    // ── Overlay ──
    const overlayStyle = {
        position: 'fixed', inset: 0, zIndex: 9999,
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column',
        overflow: 'hidden',
        background: 'linear-gradient(160deg, #021a0f 0%, #03120b 40%, #0a1628 100%)',
        opacity: isExiting ? 0 : 1,
        transition: 'opacity 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
        fontFamily: "'Inter', sans-serif",
    };

    // ── Center container ──
    const centerStyle = {
        position: 'relative',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        width: 340, height: 340,
    };

    // ── Glow behind moon ──
    const glowStyle = {
        position: 'absolute', width: '100%', height: '100%', borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.08) 40%, transparent 70%)',
        transform: isRevealed ? 'scale(1.2)' : 'scale(0.3)',
        opacity: isRevealed ? 1 : 0,
        transition: 'all 1s cubic-bezier(0.16, 1, 0.3, 1)',
        filter: 'blur(8px)',
    };

    // ── Moon icon ──
    const moonStyle = {
        fontSize: 130,
        color: fontsReady ? '#10b981' : 'transparent',
        transition: 'all 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
        transform: isRevealed ? 'scale(1) rotate(0deg)' : 'scale(0.2) rotate(-60deg)',
        opacity: isRevealed ? 1 : 0,
        filter: isRadiating
            ? 'drop-shadow(0 0 40px rgba(16, 185, 129, 0.5)) drop-shadow(0 0 80px rgba(5, 150, 105, 0.3))'
            : 'none',
        fontVariationSettings: "'FILL' 1, 'wght' 200, 'GRAD' 0, 'opsz' 48",
        zIndex: 10, position: 'relative',
    };

    // ── Bismillah text ──
    const bismillahStyle = {
        fontFamily: "'Amiri', serif",
        fontSize: 28,
        color: 'rgba(217, 171, 95, 0.85)',
        letterSpacing: '0.04em',
        marginTop: 20,
        opacity: isRadiating ? 1 : 0,
        transform: isRadiating ? 'translateY(0) scale(1)' : 'translateY(12px) scale(0.9)',
        transition: 'all 0.7s cubic-bezier(0.16, 1, 0.3, 1) 0.15s',
        textShadow: '0 0 20px rgba(217, 171, 95, 0.3)',
        zIndex: 10,
        direction: 'rtl',
    };

    // ── Title ──
    const titleStyle = {
        fontSize: 42, fontWeight: 800,
        background: 'linear-gradient(135deg, #10b981, #34d399, #d9ab5f)',
        WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
        letterSpacing: '-0.02em',
        marginTop: 14,
        opacity: isRadiating ? 1 : 0,
        transform: isRadiating ? 'translateY(0)' : 'translateY(18px)',
        transition: 'all 0.7s cubic-bezier(0.16, 1, 0.3, 1) 0.3s',
        zIndex: 10,
    };

    // ── Subtitle ──
    const subtitleStyle = {
        fontSize: 14, fontWeight: 500,
        color: 'rgba(167, 243, 208, 0.6)',
        letterSpacing: '0.25em', textTransform: 'uppercase',
        marginTop: 8,
        opacity: isRadiating ? 1 : 0,
        transform: isRadiating ? 'translateY(0)' : 'translateY(12px)',
        transition: 'all 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.5s',
        zIndex: 10,
    };

    // ── Progress bar ──
    const progressTrackStyle = {
        width: 180, height: 2, borderRadius: 4,
        background: 'rgba(16, 185, 129, 0.12)',
        marginTop: 36, overflow: 'hidden',
        opacity: isRadiating ? 1 : 0,
        transition: 'opacity 0.4s ease 0.6s', zIndex: 10,
    };
    const progressBarStyle = {
        height: '100%', borderRadius: 4,
        background: 'linear-gradient(90deg, #059669, #10b981, #d9ab5f)',
        animation: 'progressFill 3s ease-out 0.8s forwards',
        width: 0,
    };

    return h('div', { style: overlayStyle, className: 'react-intro' },

        // ── Ambient Background Glow ──
        h('div', {
            style: {
                position: 'absolute', inset: 0,
                backgroundImage: `
                    radial-gradient(ellipse at 30% 20%, rgba(5, 150, 105, 0.06) 0%, transparent 50%),
                    radial-gradient(ellipse at 70% 80%, rgba(217, 171, 95, 0.04) 0%, transparent 50%),
                    radial-gradient(ellipse at 50% 50%, rgba(16, 185, 129, 0.03) 0%, transparent 60%)
                `,
                animation: 'glowPulse 6s ease-in-out infinite',
            }
        }),

        // ── Starfield Background ──
        h(Starfield, { count: 80 }),

        // ── Floating Motes ──
        h(FloatingMotes, { count: 30, visible: isRevealed }),

        // ── Center Assembly ──
        h('div', { style: centerStyle },

            // Glow
            h('div', { style: glowStyle }),

            // Geometric rings (Islamic-inspired rotating patterns)
            h(GeometricRing, {
                size: 320, duration: 30, opacity: isRevealed ? 0.12 : 0,
                borderStyle: '1px solid rgba(16, 185, 129, 0.12)',
            }),
            h(GeometricRing, {
                size: 270, duration: 20, reverse: true, opacity: isRevealed ? 0.15 : 0,
                borderStyle: '1px dashed rgba(217, 171, 95, 0.14)',
            }),
            h(GeometricRing, {
                size: 220, duration: 35, opacity: isRevealed ? 0.1 : 0,
                borderStyle: '1px solid rgba(52, 211, 153, 0.1)',
            }),
            h(GeometricRing, {
                size: 180, duration: 15, reverse: true, opacity: isRevealed ? 0.08 : 0,
                borderStyle: '1px dotted rgba(217, 171, 95, 0.1)',
            }),

            // 8-Pointed Islamic Star (Rub el Hizb inspired)
            h('div', {
                style: {
                    position: 'absolute',
                    width: 200, height: 200,
                    top: '50%', left: '50%',
                    transform: 'translate(-50%, -50%)',
                    opacity: isRevealed ? 0.12 : 0,
                    transition: 'opacity 1.5s ease 0.3s',
                    animation: isRevealed ? 'orbit 40s linear infinite reverse' : 'none',
                },
            },
                h('svg', { viewBox: '0 0 100 100', width: '100%', height: '100%' },
                    // Two overlapping rotated squares → 8-pointed star
                    h('rect', {
                        x: 22, y: 22, width: 56, height: 56,
                        fill: 'none', stroke: 'rgba(217, 171, 95, 0.35)', strokeWidth: 0.8,
                        transform: 'rotate(0 50 50)',
                    }),
                    h('rect', {
                        x: 22, y: 22, width: 56, height: 56,
                        fill: 'none', stroke: 'rgba(16, 185, 129, 0.3)', strokeWidth: 0.8,
                        transform: 'rotate(45 50 50)',
                    })
                )
            ),

            // Light rays
            h(LightRays, { visible: isRadiating }),

            // Pulsing halos
            h(PulsingHalos, { visible: isRadiating }),

            // Moon icon
            h('span', {
                className: 'material-symbols-outlined',
                style: { ...moonStyle, fontSize: 140 },
            }, 'dark_mode'),
        ),

        // ── Text & Branding ──
        h('div', { style: { textAlign: 'center', position: 'relative', zIndex: 10 } },
            h('div', { style: bismillahStyle }, 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ'),
            h('div', { style: titleStyle }, 'Al Huda'),
            h('div', { style: subtitleStyle }, 'Islamic Knowledge Assistant'),
        ),

        // ── Progress Bar ──
        h('div', { style: progressTrackStyle },
            h('div', { style: progressBarStyle })
        )
    );
}

// ─────────────────────────────
// Mount
// ─────────────────────────────
function mountReactIntro() {
    const container = document.getElementById('reactIntroRoot');
    if (!container) return;

    const animationEnabled = localStorage.getItem('launchAnimation') !== 'false';
    if (!animationEnabled) {
        document.body.style.opacity = '1';
        document.body.classList.add('fonts-loaded');
        return;
    }

    const root = ReactDOM.createRoot(container);
    root.render(h(IntroAnimation, {
        onComplete: () => {
            const welcomeElements = document.querySelectorAll('#welcomeScreen > *');
            welcomeElements.forEach((el, index) => {
                el.classList.add('content-reveal', 'reveal-delay-' + Math.min(index + 1, 5));
            });
            setTimeout(() => root.unmount(), 100);
        }
    }));
}

window.mountReactIntro = mountReactIntro;
