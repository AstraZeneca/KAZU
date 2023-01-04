import {LSComponent} from "./ls-component";
import {TextSubmit} from "./text-submit";
import React from "react";
import {KazuResponse} from "../types/types";
import {IKazuClient} from "../utils/kazu-client";
import { JsonViewer } from '@textea/json-viewer'

type AppProps = {
    kazuApiClient: IKazuClient;
    authEnabled: boolean;
}

type AppState = {
    text?: string;
    text_ner_result?: KazuResponse;
}

class App extends React.Component<AppProps, AppState> {
    constructor(props: AppProps) {
        super(props);
        this.state = {
            text: undefined,
            text_ner_result: undefined,
        }
    }

    setKazuResponse(kazuResponse: KazuResponse) {
        this.setState((state) => ({
            text_ner_result: kazuResponse,
        }))
    }

    render() {
        const kazuResp = this.state.text_ner_result;
        let lsComponent;
        let jsonViewComponent;
        if (kazuResp !== undefined) {
            lsComponent = <LSComponent kazuWebDocument={kazuResp.parsedDocument}/>
            jsonViewComponent = <JsonViewer value={kazuResp.rawDocument}/>
        } else {
            lsComponent = undefined
            jsonViewComponent = undefined
        }

        return (
            <div className="app-root">
                <TextSubmit ner_response_callback={this.setKazuResponse.bind(this)} kazu_client={this.props.kazuApiClient} auth_enabled={this.props.authEnabled}/>
                {lsComponent}
                {jsonViewComponent}
            </div>
        )
    }
}

export {App}
