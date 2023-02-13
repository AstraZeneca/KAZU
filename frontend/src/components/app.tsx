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
        this.setState((_) => ({
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

        const appTitleText = (
            <div className={"app-title-text"}>
                <span>KAZU Named Entity Recognition</span>
            </div>
        )

        const kazuBlurb = (
            <div className={"kazu-blurb"}><span><strong>KAZU</strong> is our next gen NLP framework, specifically designed to deal with
                the complexities of BioNER. It's designed to be production grade, resilient, fast, scalable, and easy to
                extend, allowing you to quickly deploy emergent state-of-the-art NLP methods.

                For more information and support, please visit <a href={"https://astrazeneca.github.com/kazu"}>https://astrazeneca.github.com/kazu</a>
            </span>
            </div>
        )
        return (
            <div className="app-root">
                {appTitleText}
                <TextSubmit ner_response_callback={this.setKazuResponse.bind(this)} kazu_client={this.props.kazuApiClient} auth_enabled={this.props.authEnabled}/>
                {kazuBlurb}
                {lsComponent}
                {jsonViewComponent}
            </div>
        )
    }
}

export {App}
