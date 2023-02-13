import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import {App} from './components/app';
import {KazuClient} from "./utils/kazu-client";
import {Config} from "./config";

const conf = new Config();

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

const kazuClient = new KazuClient(conf.kazuApiUrl())
root.render(
    <App kazuApiClient={kazuClient} authEnabled={conf.authEnabled()}/>
);
