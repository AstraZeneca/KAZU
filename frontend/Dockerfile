FROM nginx:1.23.3-alpine

ENV APP_ROOT=/opt/app
ENV KAZU_API_URL=NOT_SET

RUN mkdir $APP_ROOT
WORKDIR $APP_ROOT

RUN apk update
RUN apk add nodejs npm jq

COPY package.json ./package.json
COPY tsconfig.json ./tsconfig.json
COPY src ./src
COPY public ./public

COPY nginx.template /etc/nginx/templates/nginx.template


RUN npm install --omit dev

COPY start-server.sh ./start-server.sh
RUN chmod u+x ./start-server.sh
CMD ["./start-server.sh"]
