# 🔐 Google OAuth Setup Guide for CodeEx AI

## 📋 **Step-by-Step Setup Instructions**

### 1. **Create Google Cloud Project**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Create Project" or select an existing project
3. Name your project (e.g., "CodeEx AI")
4. Click "Create"

### 2. **Enable Google+ API**

1. In the Google Cloud Console, go to "APIs & Services" > "Library"
2. Search for "Google+ API" 
3. Click on it and press "Enable"
4. Also enable "Google Identity" API

### 3. **Create OAuth 2.0 Credentials**

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. If prompted, configure the OAuth consent screen first:
   - Choose "External" for user type
   - Fill in required fields:
     - App name: "CodeEx AI"
     - User support email: your email
     - Developer contact: your email
   - Add scopes: `email`, `profile`, `openid`
   - Add test users (your email addresses)

4. Create OAuth 2.0 Client ID:
   - Application type: "Web application"
   - Name: "CodeEx AI Web Client"
   - Authorized JavaScript origins:
     - `http://localhost:5000`
     - `http://127.0.0.1:5000`
     - Add your production domain when ready
   - Authorized redirect URIs:
     - `http://localhost:5000/auth/google/callback`
     - `http://127.0.0.1:5000/auth/google/callback`
     - Add your production callback URL when ready

5. Click "Create"
6. **Copy the Client ID and Client Secret** - you'll need these!

### 4. **Configure Environment Variables**

1. Open your `.env` file in the project root
2. Replace the placeholder values:

```env
# Google OAuth Configuration
GOOGLE_CLIENT_ID=your-actual-client-id-here.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-actual-client-secret-here
```

**Example:**
```env
GOOGLE_CLIENT_ID=123456789-abcdefghijklmnop.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-abcdefghijklmnopqrstuvwxyz
```

### 5. **Test the Setup**

1. Restart your Flask application:
   ```bash
   python app.py
   ```

2. Go to `http://localhost:5000/login`
3. Click "Continue with Google"
4. You should see the Google OAuth popup

## 🔧 **Troubleshooting Common Issues**

### Issue 1: "Error 400: redirect_uri_mismatch"
**Solution:** Make sure your redirect URIs in Google Console match exactly:
- `http://localhost:5000/auth/google/callback`
- `http://127.0.0.1:5000/auth/google/callback`

### Issue 2: "Error 403: access_blocked"
**Solution:** 
- Make sure your OAuth consent screen is configured
- Add your email as a test user
- Verify your app is not in "Testing" mode for production

### Issue 3: Google button not appearing
**Solution:**
- Check browser console for JavaScript errors
- Verify Google Client ID is correctly set in `.env`
- Make sure the Google API script is loading

### Issue 4: "Invalid client ID"
**Solution:**
- Double-check your Client ID in `.env` file
- Make sure there are no extra spaces or characters
- Verify the Client ID ends with `.apps.googleusercontent.com`

## 🚀 **Production Deployment**

When deploying to production:

1. **Update OAuth Settings:**
   - Add your production domain to "Authorized JavaScript origins"
   - Add your production callback URL to "Authorized redirect URIs"

2. **Update Environment Variables:**
   ```env
   GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=your-client-secret
   ```

3. **Publish OAuth Consent Screen:**
   - Go to "OAuth consent screen" in Google Console
   - Click "Publish App" to make it available to all users

## 📝 **Security Best Practices**

1. **Never commit credentials to version control**
2. **Use environment variables for all sensitive data**
3. **Regularly rotate your client secrets**
4. **Monitor OAuth usage in Google Console**
5. **Use HTTPS in production**

## 🧪 **Testing Your Setup**

Run this test to verify your Google OAuth is working:

```bash
# Check if environment variables are loaded
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Google Client ID:', os.getenv('GOOGLE_CLIENT_ID'))
print('Google Client Secret:', 'Set' if os.getenv('GOOGLE_CLIENT_SECRET') else 'Not Set')
"
```

## 📞 **Need Help?**

If you're still having issues:

1. Check the browser console for JavaScript errors
2. Check the Flask application logs
3. Verify your Google Cloud Console settings
4. Make sure all URLs match exactly (including http/https)

## ✅ **Verification Checklist**

- [ ] Google Cloud Project created
- [ ] Google+ API and Google Identity API enabled
- [ ] OAuth consent screen configured
- [ ] OAuth 2.0 Client ID created
- [ ] Authorized origins and redirect URIs added
- [ ] Client ID and Secret added to `.env` file
- [ ] Flask app restarted
- [ ] Google login button appears on login page
- [ ] Google OAuth popup works
- [ ] User can successfully log in with Google

---

**🎉 Once completed, your CodeEx AI will have fully functional Google OAuth authentication!**