/* ─── Navbar scroll effect ───────────────────────────────────────── */
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 60);
}, { passive: true });

/* ─── Mobile menu ────────────────────────────────────────────────── */
const hamburger = document.getElementById('hamburger');
const mobileMenu = document.getElementById('mobileMenu');
const mobileClose = document.getElementById('mobileClose');
const mobileLinks = document.querySelectorAll('.mobile-link');

hamburger.addEventListener('click', () => mobileMenu.classList.add('open'));
mobileClose.addEventListener('click', () => mobileMenu.classList.remove('open'));
mobileLinks.forEach(link => link.addEventListener('click', () => mobileMenu.classList.remove('open')));

/* ─── Countdown timer ────────────────────────────────────────────── */
const weddingDate = new Date('2026-06-14T16:30:00');

function updateCountdown() {
  const now = new Date();
  const diff = weddingDate - now;

  if (diff <= 0) {
    document.getElementById('countdown').innerHTML = '<p style="font-family:var(--font-serif);font-size:2rem;">Today is the day!</p>';
    return;
  }

  const days    = Math.floor(diff / (1000 * 60 * 60 * 24));
  const hours   = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  const seconds = Math.floor((diff % (1000 * 60)) / 1000);

  document.getElementById('days').textContent    = String(days).padStart(2, '0');
  document.getElementById('hours').textContent   = String(hours).padStart(2, '0');
  document.getElementById('minutes').textContent = String(minutes).padStart(2, '0');
  document.getElementById('seconds').textContent = String(seconds).padStart(2, '0');
}

updateCountdown();
setInterval(updateCountdown, 1000);

/* ─── FAQ accordion ──────────────────────────────────────────────── */
document.querySelectorAll('.faq-question').forEach(btn => {
  btn.addEventListener('click', () => {
    const answer = btn.nextElementSibling;
    const isOpen = btn.getAttribute('aria-expanded') === 'true';

    // Close all others
    document.querySelectorAll('.faq-question').forEach(other => {
      if (other !== btn) {
        other.setAttribute('aria-expanded', 'false');
        other.nextElementSibling.classList.remove('open');
      }
    });

    btn.setAttribute('aria-expanded', String(!isOpen));
    answer.classList.toggle('open', !isOpen);
  });
});

/* ─── RSVP form ──────────────────────────────────────────────────── */
const form = document.getElementById('rsvpForm');
const guestCountGroup = document.getElementById('guestCountGroup');
const mealGroup = document.getElementById('mealGroup');
const dietaryGroup = document.getElementById('dietaryGroup');

// Show/hide guest details based on attendance
document.querySelectorAll('input[name="attending"]').forEach(radio => {
  radio.addEventListener('change', () => {
    const attending = radio.value === 'yes';
    guestCountGroup.style.display = attending ? '' : 'none';
    mealGroup.style.display       = attending ? '' : 'none';
    dietaryGroup.style.display    = attending ? '' : 'none';
  });
});

function setError(fieldId, message) {
  const field = document.getElementById(fieldId);
  const error = document.getElementById(fieldId + 'Error');
  if (field) field.classList.toggle('invalid', !!message);
  if (error) error.textContent = message || '';
}

function validateEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

form.addEventListener('submit', e => {
  e.preventDefault();

  const firstName  = form.firstName.value.trim();
  const lastName   = form.lastName.value.trim();
  const email      = form.email.value.trim();
  const attending  = form.querySelector('input[name="attending"]:checked');

  let valid = true;

  if (!firstName) { setError('firstName', 'Please enter your first name.'); valid = false; }
  else setError('firstName', '');

  if (!lastName) { setError('lastName', 'Please enter your last name.'); valid = false; }
  else setError('lastName', '');

  if (!email || !validateEmail(email)) { setError('email', 'Please enter a valid email address.'); valid = false; }
  else setError('email', '');

  const attendingError = document.getElementById('attendingError');
  if (!attending) { attendingError.textContent = 'Please let us know if you can make it.'; valid = false; }
  else attendingError.textContent = '';

  if (!valid) return;

  // In a real site, submit via fetch() to a backend / form service here.
  // For now, show a success message.
  form.querySelectorAll('input, select, textarea, button[type="submit"]').forEach(el => el.disabled = true);
  document.getElementById('successName').textContent = firstName;
  document.getElementById('formSuccess').classList.remove('hidden');
  form.querySelector('.submit-btn').style.display = 'none';
});

/* ─── Scroll fade-in animations ──────────────────────────────────── */
const animateTargets = [
  '.section-title', '.timeline-item', '.event-card',
  '.party-member', '.gallery-item', '.travel-card',
  '.registry-card', '.faq-item', '.section-sub'
];

animateTargets.forEach(selector => {
  document.querySelectorAll(selector).forEach(el => el.classList.add('fade-in'));
});

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });

document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

/* ─── Smooth active nav highlighting ─────────────────────────────── */
const sections = document.querySelectorAll('section[id]');
const navAnchors = document.querySelectorAll('.nav-links a');

const sectionObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      navAnchors.forEach(a => a.style.opacity = '');
      const active = document.querySelector(`.nav-links a[href="#${entry.target.id}"]`);
      if (active) active.style.fontWeight = '400';
    }
  });
}, { rootMargin: '-40% 0px -55% 0px' });

sections.forEach(s => sectionObserver.observe(s));
