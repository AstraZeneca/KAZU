import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import {App} from './components/app';
import reportWebVitals from './reportWebVitals';
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

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
