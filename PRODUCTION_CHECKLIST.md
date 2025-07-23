# CodeEx AI Production Deployment Checklist

## Pre-Deployment Checklist

### ✅ Environment Configuration
- [ ] Set `FLASK_DEBUG=false` in production
- [ ] Generate strong `SECRET_KEY` (32+ characters)
- [ ] Configure `FORCE_HTTPS=true`
- [ ] Set proper `ALLOWED_ORIGINS` for CORS
- [ ] Verify all API keys are set (OpenAI, Google Gemini)
- [ ] Configure email settings for contact form
- [ ] Set up Google OAuth credentials

### ✅ Security
- [ ] Install security packages: `flask-limiter`, `flask-talisman`
- [ ] Enable rate limiting on sensitive endpoints
- [ ] Configure Content Security Policy (CSP)
- [ ] Set up HTTPS/SSL certificates
- [ ] Enable security headers
- [ ] Review and test authentication flows
- [ ] Verify password reset functionality

### ✅ Database
- [ ] Initialize production database
- [ ] Set up database backups
- [ ] Configure proper database permissions
- [ ] Test database connection pooling
- [ ] Verify encryption for sensitive data

### ✅ Dependencies
- [ ] Install all required packages from requirements.txt
- [ ] Update packages to latest secure versions
- [ ] Remove development-only dependencies
- [ ] Verify AI model dependencies are working

### ✅ Performance
- [ ] Configure Gunicorn with appropriate worker count
- [ ] Set up reverse proxy (Nginx recommended)
- [ ] Enable gzip compression
- [ ] Configure static file caching
- [ ] Test application under load

## Deployment Steps

### 1. Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3 python3-pip python3-venv nginx -y

# Create application user
sudo useradd -m -s /bin/bash codeex
sudo usermod -aG www-data codeex
```

### 2. Application Deployment
```bash
# Clone repository
git clone <your-repo-url> /var/www/codeex-ai
cd /var/www/codeex-ai

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Run deployment script
python deploy.py
```

### 3. Web Server Configuration
```bash
# Copy nginx configuration
sudo cp nginx-codeex-ai.conf /etc/nginx/sites-available/codeex-ai
sudo ln -s /etc/nginx/sites-available/codeex-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Service Management
```bash
# Enable and start systemd service
sudo systemctl enable codeex-ai
sudo systemctl start codeex-ai
sudo systemctl status codeex-ai
```

## Post-Deployment Checklist

### ✅ Functionality Testing
- [ ] Test user registration and login
- [ ] Verify Google OAuth authentication
- [ ] Test AI chat functionality with all providers
- [ ] Verify email sending (contact form, password reset)
- [ ] Test file upload/download features
- [ ] Verify PWA functionality
- [ ] Test offline capabilities

### ✅ Security Testing
- [ ] Verify HTTPS is working and enforced
- [ ] Test rate limiting on login attempts
- [ ] Verify CSRF protection is active
- [ ] Test XSS protection headers
- [ ] Verify secure cookie settings
- [ ] Test password reset security

### ✅ Performance Testing
- [ ] Load test with multiple concurrent users
- [ ] Monitor response times
- [ ] Check memory usage under load
- [ ] Verify database performance
- [ ] Test AI response times

### ✅ Monitoring Setup
- [ ] Set up application logging
- [ ] Configure log rotation
- [ ] Set up error monitoring (optional: Sentry)
- [ ] Monitor system resources
- [ ] Set up uptime monitoring
- [ ] Configure backup schedules

## Environment Variables Reference

### Required Variables
```bash
SECRET_KEY=your-super-secret-key-here
FLASK_DEBUG=false
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
OPENAI_API_KEY=your-openai-api-key
GOOGLE_GEMINI_API_KEY=your-gemini-api-key
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
CONTACT_EMAIL=the.heoster@mail.com
```

### Optional Variables
```bash
HOST=0.0.0.0
PORT=5000
FORCE_HTTPS=true
ALLOWED_ORIGINS=https://yourdomain.com
DATABASE_URL=codeex.db
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
```

## Maintenance Tasks

### Daily
- [ ] Check application logs for errors
- [ ] Monitor system resources
- [ ] Verify backup completion

### Weekly
- [ ] Review security logs
- [ ] Update dependencies if needed
- [ ] Check AI performance metrics
- [ ] Clean up old log files

### Monthly
- [ ] Security audit
- [ ] Performance review
- [ ] Database optimization
- [ ] Backup testing

## Troubleshooting

### Common Issues

1. **Application won't start**
   - Check environment variables
   - Verify database permissions
   - Check log files in `/var/log/`

2. **AI providers not working**
   - Verify API keys are correct
   - Check network connectivity
   - Review rate limits

3. **Email not sending**
   - Verify SMTP settings
   - Check app password for Gmail
   - Review firewall settings

4. **Performance issues**
   - Increase Gunicorn workers
   - Check database queries
   - Monitor memory usage

### Log Locations
- Application logs: `app.log`
- Nginx logs: `/var/log/nginx/`
- System logs: `/var/log/syslog`
- Service logs: `journalctl -u codeex-ai`

## Security Considerations

1. **Keep secrets secure**
   - Never commit API keys to version control
   - Use environment variables or secret management
   - Rotate keys regularly

2. **Regular updates**
   - Keep Python packages updated
   - Update system packages
   - Monitor security advisories

3. **Access control**
   - Use proper file permissions
   - Limit database access
   - Monitor failed login attempts

4. **Backup strategy**
   - Regular database backups
   - Test backup restoration
   - Store backups securely

## Support

For issues or questions:
- Check the application logs first
- Review this checklist
- Contact: the.heoster@mail.com

---

**Last updated:** January 2025
**Version:** 1.0