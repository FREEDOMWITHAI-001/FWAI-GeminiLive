"""
WhatsApp Message Templates
Beautiful, formatted templates that AI can use during calls
Based on conversation context, AI selects the appropriate template
"""

WHATSAPP_TEMPLATES = {
    # =========================================================================
    # TEMPLATE 1: COURSE/PRODUCT INFORMATION
    # Use when: User asks about courses, products, features, what's included
    # =========================================================================
    "course_details": {
        "name": "Course/Product Information",
        "description": "Send when user asks about courses, products, features, pricing, or what's included",
        "template": """
╔══════════════════════════════════════╗
       ✨ *{course_name}* ✨
╚══════════════════════════════════════╝

Hello {customer_name}! 👋

Here's everything you need to know:

┌─────────────────────────────────────┐
│  📚  *What's Included*              │
├─────────────────────────────────────┤
│                                     │
│  ✓  Complete video course library   │
│  ✓  Hands-on projects & exercises   │
│  ✓  Weekly live mentorship calls    │
│  ✓  Private community access        │
│  ✓  Lifetime updates & resources    │
│  ✓  Certificate of completion       │
│                                     │
└─────────────────────────────────────┘

💰 *Investment:* {price}

🎁 *Special Bonus:*
   Free 1-on-1 onboarding session

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔗 *Learn More:* {enrollment_link}

Have questions? Just reply! 💬
I'm here to help.

_Sent with ❤️ from {company_name}_
"""
    },

    # =========================================================================
    # TEMPLATE 2: PAYMENT/PURCHASE LINK
    # Use when: User is ready to buy, asks for payment, wants to enroll
    # =========================================================================
    "payment_link": {
        "name": "Payment/Purchase Link",
        "description": "Send when user is ready to purchase, asks for payment options, or wants to enroll now",
        "template": """
╔══════════════════════════════════════╗
     🎉 *Ready to Get Started!* 🎉
╚══════════════════════════════════════╝

Hey {customer_name}!

Great choice! Let's get you enrolled. 🚀

┌─────────────────────────────────────┐
│  📦  *Your Order Summary*           │
├─────────────────────────────────────┤
│                                     │
│  Course:  {course_name}             │
│  Amount:  {price}                   │
│                                     │
└─────────────────────────────────────┘

🔐 *Secure Payment Link:*
👉 {payment_link}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ *What happens next:*

   1️⃣  Complete secure payment
   2️⃣  Instant access to course
   3️⃣  Welcome email with login
   4️⃣  Onboarding call scheduled

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🛡️ *100% Secure* | 💳 *All Cards Accepted*
📅 *7-Day Money-Back Guarantee*

Need help? Just reply here! 🙋

_Sent with ❤️ from {company_name}_
"""
    },

    # =========================================================================
    # TEMPLATE 3: SUPPORT/HELP
    # Use when: User has issues, complaints, technical problems, needs help
    # =========================================================================
    "support_contact": {
        "name": "Support/Help Contact",
        "description": "Send when user has issues, complaints, technical problems, or needs customer support",
        "template": """
╔══════════════════════════════════════╗
      🤝 *We're Here For You* 🤝
╚══════════════════════════════════════╝

Hi {customer_name}!

I understand you need some help.
Don't worry, we've got you covered! 💪

┌─────────────────────────────────────┐
│  📞  *How to Reach Us*              │
├─────────────────────────────────────┤
│                                     │
│  💬  *WhatsApp* (Fastest!)          │
│      Reply to this message          │
│      ⏰ Response: < 2 hours         │
│                                     │
│  📧  *Email*                        │
│      {support_email}                │
│      ⏰ Response: < 24 hours        │
│                                     │
│  📚  *Help Center*                  │
│      {help_link}                    │
│      ⏰ 24/7 self-service           │
│                                     │
└─────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌟 *Our Promise:*

   ✓  We take every concern seriously
   ✓  Quick resolution is our priority
   ✓  Your satisfaction matters to us

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Just describe your issue and we'll
sort it out together! 🙌

_Sent with ❤️ from {company_name}_
"""
    }
}


def get_template(template_id: str) -> dict:
    """Get a template by ID"""
    return WHATSAPP_TEMPLATES.get(template_id)


def format_template(template_id: str, context: dict) -> str:
    """Format a template with context variables"""
    template = WHATSAPP_TEMPLATES.get(template_id)
    if not template:
        return None

    message = template["template"].strip()

    # Set defaults for missing context values
    defaults = {
        "customer_name": "there",
        "course_name": "Our Premium Course",
        "price": "Contact for pricing",
        "enrollment_link": "Contact us for details",
        "payment_link": "Contact us for payment link",
        "support_email": "support@freedomwithai.com",
        "help_link": "https://freedomwithai.com/help"
    }

    # Merge defaults with provided context
    full_context = {**defaults, **context}

    # Replace all placeholders with context values
    for key, value in full_context.items():
        message = message.replace(f"{{{key}}}", str(value))

    return message


def get_all_template_names() -> list:
    """Get list of all template names for AI tool description"""
    return [
        f"{tid}: {t['description']}"
        for tid, t in WHATSAPP_TEMPLATES.items()
    ]


def get_template_descriptions() -> str:
    """Get formatted template descriptions for AI prompt"""
    descriptions = []
    for tid, t in WHATSAPP_TEMPLATES.items():
        descriptions.append(f"- {tid}: {t['description']}")
    return "\n".join(descriptions)
