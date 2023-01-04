import React from "react";
import {KazuResponse} from "../types/types";
import {IKazuClient} from "../utils/kazu-client";

type TextSubmitProps = {
    ner_response_callback: (arg: KazuResponse) => void;
    kazu_client: IKazuClient
    auth_enabled: boolean
}

type TextSubmitState = {
    textAreaValue: string
}

class TextSubmit extends React.Component<TextSubmitProps, TextSubmitState> {
    constructor(props: any) {
        super(props);
        this.state = {
            textAreaValue: "EGFR"
        }
    }

    handleButtonClick() {
        const kazuClient = this.props.kazu_client;
        const text = this.state.textAreaValue;
        kazuClient.ner(text).then(this.props.ner_response_callback);
    }

    handleTextAreaChange(event: React.ChangeEvent<HTMLTextAreaElement>) {
        this.setState((_) => {
            return {textAreaValue: event.target.value};
        })
    }

    render() {
        let authInputComponent;
        if(this.props.auth_enabled) {
            authInputComponent = (
                <div>
                    <textarea id={"authInput"} placeholder={"JWT token"}></textarea>
                </div>
            )
        } else {
            authInputComponent = undefined
        }
        return (
            <div id={"textWidget"}>
                <textarea value={this.state.textAreaValue} onChange={this.handleTextAreaChange.bind(this)}/>
                {authInputComponent}
                <button onClick={this.handleButtonClick.bind(this)}>Submit</button>
            </div>
        );
    }
}

export {TextSubmit}
