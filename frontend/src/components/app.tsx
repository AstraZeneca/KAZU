import {LSComponent} from "./ls-component";
import {TextSubmit} from "./text-submit";
import React from "react";
import {KazuLSResponse} from "../types/types";
import {IKazuClient} from "../utils/kazu-client";
import {JsonView} from "./json-view";

type AppProps = {
    kazuApiClient: IKazuClient;
    authEnabled: boolean;
}

type AppState = {
    text?: string;
    text_ner_result?: KazuLSResponse;
}

class App extends React.Component<AppProps, AppState> {
    constructor(props: AppProps) {
        super(props);
        this.state = {
            text: undefined,
            text_ner_result: undefined,
        }
    }

    setKazuResponse(kazuResponse: KazuLSResponse) {
        this.setState((state) => ({
            text_ner_result: kazuResponse,
        }))
    }

    render() {
        const kazuResp = this.state.text_ner_result;
        let lsComponent;
        let jsonViewComponent;
        if (kazuResp !== undefined) {
            lsComponent = <LSComponent kazuLSAnnotations={kazuResp}/>
            jsonViewComponent = <JsonView rawDocument={kazuResp.rawDocument}/>
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
