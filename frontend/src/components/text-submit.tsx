import React from "react";
import {KazuLSResponse} from "../types/types";
import {IKazuClient} from "../utils/kazu-client";

type TextSubmitProps = {
    ner_response_callback: (arg: KazuLSResponse) => void;
    kazu_client: IKazuClient
    auth_enabled: boolean
}

type TextSubmitState = {
    textAreaValue: string;
    authAreaValue?: string
}

class TextSubmit extends React.Component<TextSubmitProps, TextSubmitState> {
    constructor(props: any) {
        super(props);
        this.state = {
            textAreaValue: "EGFR",
            authAreaValue: undefined
        }
    }

    handleButtonClick() {
        const kazuClient = this.props.kazu_client;
        const text = this.state.textAreaValue;
        const auth = this.state.authAreaValue;
        kazuClient.ner_with_ls(text, auth).then(this.props.ner_response_callback);
    }

    handleTextAreaChange(event: React.ChangeEvent<HTMLTextAreaElement>) {
        this.setState((_) => {
            return {textAreaValue: event.target.value};
        })
    }

    handleAuthAreaChange(event: React.ChangeEvent<HTMLTextAreaElement>) {
        this.setState((_) => {
            return {authAreaValue: event.target.value};
        })
    }

    render() {
        let authInputComponent;
        if(this.props.auth_enabled) {
            authInputComponent = (
                <div>
                    <textarea id={"authInput"} placeholder={"JWT token"} onChange={this.handleAuthAreaChange.bind(this)}/>
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
