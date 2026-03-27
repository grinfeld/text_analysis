FROM nginx:alpine

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy our static files
COPY static/ /usr/share/nginx/html/

# Custom nginx config: serve static files + proxy /sentiment and /metrics to backend
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
