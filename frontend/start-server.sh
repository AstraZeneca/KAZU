jq '.kazuApiUrl = "'$KAZU_API_URL'"' src/config.json > src/config.tmp.json
cp src/config.tmp.json src/config.json

# configure the nginx template
envsubst '\$APP_ROOT,\$KAZU_API_URL' < /etc/nginx/templates/nginx.template > /etc/nginx/nginx.conf

npm run build
cp -r build "$APP_ROOT"/build
/usr/sbin/nginx -g 'daemon off;'
