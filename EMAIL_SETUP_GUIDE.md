# 📧 Email Setup Guide for Contact Form

## 🎯 **Contact Form Email Forwarding**

All contact form messages are automatically forwarded to: **the.heoster@mail.com**

## ⚙️ **Email Configuration Options**

### Option 1: Gmail SMTP (Recommended)

1. **Create a Gmail App Password:**
   - Go to your Google Account settings
   - Navigate to Security → 2-Step Verification
   - Scroll down to "App passwords"
   - Generate a new app password for "Mail"
   - Copy the 16-character password

2. **Update .env file:**
   ```env
   EMAIL_HOST=smtp.gmail.com
   EMAIL_PORT=587
   EMAIL_USERNAME=your-gmail@gmail.com
   EMAIL_PASSWORD=your-16-character-app-password
   ```

### Option 2: Other Email Providers

**Outlook/Hotmail:**
```env
EMAIL_HOST=smtp-mail.outlook.com
EMAIL_PORT=587
EMAIL_USERNAME=your-email@outlook.com
EMAIL_PASSWORD=your-password
```

**Yahoo Mail:**
```env
EMAIL_HOST=smtp.mail.yahoo.com
EMAIL_PORT=587
EMAIL_USERNAME=your-email@yahoo.com
EMAIL_PASSWORD=your-app-password
```

**Custom SMTP Server:**
```env
EMAIL_HOST=your-smtp-server.com
EMAIL_PORT=587
EMAIL_USERNAME=your-email@yourdomain.com
EMAIL_PASSWORD=your-password
```

## 🔧 **Setup Instructions**

1. **Choose your email provider** (Gmail recommended)

2. **Update your .env file** with the correct credentials:
   ```env
   EMAIL_USERNAME=your-actual-email@gmail.com
   EMAIL_PASSWORD=your-actual-app-password
   ```

3. **Restart your Flask application:**
   ```bash
   python app.py
   ```

4. **Test the contact form:**
   - Go to `http://localhost:5000/contact`
   - Fill out and submit the form
   - Check that the email arrives at **the.heoster@mail.com**

## 📋 **Email Template**

When someone submits the contact form, **the.heoster@mail.com** will receive an email like this:

```
Subject: [CodeEx AI Contact] Technical Support

New contact form submission from CodeEx AI:

Name: John Doe
Email: john.doe@example.com
Subject: Technical Support

Message:
I'm having trouble with the Google login feature. 
Can you help me troubleshoot this issue?

---
This message was sent from the CodeEx AI contact form.
Reply directly to this email to respond to the user.
```

## 🛡️ **Security Features**

- ✅ **Privacy Validation**: Users must agree to privacy policy
- ✅ **Form Validation**: All fields are required
- ✅ **Error Handling**: Graceful error handling with user feedback
- ✅ **Logging**: All submissions are logged for tracking
- ✅ **Spam Protection**: Form includes basic spam protection

## 📊 **Contact Categories**

The contact form includes these categories:
- **General Inquiry**: General questions about CodeEx AI
- **Technical Support**: Issues with login, features, or functionality
- **Bug Report**: Found a bug? Help improve CodeEx AI
- **Feature Request**: Suggest new features or improvements
- **Business Partnership**: Partnerships, licensing, enterprise solutions
- **Privacy & Security**: Privacy-related questions and concerns
- **Other**: Any other inquiries

## 🔍 **Testing Your Setup**

Run this test to verify email configuration:

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Email Host:', os.getenv('EMAIL_HOST'))
print('Email Port:', os.getenv('EMAIL_PORT'))
print('Email Username:', os.getenv('EMAIL_USERNAME'))
print('Email Password:', 'Set' if os.getenv('EMAIL_PASSWORD') else 'Not Set')
print('Contact Email: the.heoster@mail.com')
"
```

## 🚨 **Troubleshooting**

### Issue: "Authentication failed"
**Solution:** 
- Make sure you're using an App Password, not your regular password
- Enable 2-Factor Authentication on your Gmail account
- Generate a new App Password specifically for this application

### Issue: "Connection refused"
**Solution:**
- Check your EMAIL_HOST and EMAIL_PORT settings
- Ensure your firewall allows outbound connections on port 587
- Try using port 465 with SSL instead of 587 with TLS

### Issue: "No email received"
**Solution:**
- Check spam/junk folder at the.heoster@mail.com
- Verify the EMAIL_USERNAME is correct
- Check Flask application logs for error messages

## 📝 **Fallback Behavior**

If email configuration is not set up:
- ✅ Contact form still works
- ✅ Messages are logged in Flask application logs
- ✅ Users receive success confirmation
- ✅ You can check logs to see submitted messages

## 🎉 **Ready to Go!**

Once configured, all contact form submissions will be automatically forwarded to **the.heoster@mail.com** with:
- ✅ User's name and email for easy replies
- ✅ Subject line with category
- ✅ Full message content
- ✅ Professional email formatting
- ✅ Clear identification as CodeEx AI contact form

---

**📧 All contact form messages will be sent to: the.heoster@mail.com**