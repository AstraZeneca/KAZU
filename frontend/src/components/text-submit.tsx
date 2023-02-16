import React from "react";
import {KazuLSResponse} from "../types/types";
import {IKazuClient} from "../utils/kazu-client";
import {AxiosError} from "axios";

type TextSubmitProps = {
    ner_response_callback: (arg: KazuLSResponse) => void;
    kazu_client: IKazuClient
    auth_enabled: boolean
}

type TextSubmitState = {
    textAreaValue: string;
    authAreaValue?: string;
    showJwtTextArea: boolean;
    authenticationError: boolean;
    generalApiError: boolean
}

class TextSubmit extends React.Component<TextSubmitProps, TextSubmitState> {
    constructor(props: any) {
        super(props);
        this.state = {
            textAreaValue: "EGFR",
            authAreaValue: undefined,
            showJwtTextArea: true,
            authenticationError: false,
            generalApiError: false
        }
    }

    handleButtonClick() {
        const kazuClient = this.props.kazu_client;
        const text = this.state.textAreaValue;
        const auth = this.state.authAreaValue;
        kazuClient.ner_with_ls(text, auth)
            .then((kazuLSResp) => Promise.resolve(this.props.ner_response_callback(kazuLSResp)))
            .then(() => this.setState((_) => {
                return {
                    showJwtTextArea: false
                }
            }))
            .catch((err) => {
                console.log(err)
                if (err instanceof AxiosError && err.response) {
                    const statusCode = err.response.status;
                    if(statusCode === 401) {
                        this.setState((_) => {
                            return {authenticationError: true}
                        })
                    } else {
                        this.setState((_) => {
                            return {generalApiError: true}
                        })
                    }
                } else if(err instanceof AxiosError) {
                    this.setState((_) => {
                        return {generalApiError: true}
                    })
                }
            });
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
        let errorBox;
        if(this.props.auth_enabled) {
            if(this.state.showJwtTextArea) {
                authInputComponent = (
                    <div>
                        <textarea id={"authInput"} placeholder={"JWT token"} onChange={this.handleAuthAreaChange.bind(this)}/>
                    </div>
                )
            } else {
                authInputComponent = (
                    <div>
                        <button id={"showAuthInputButton"}>Show JWT</button>
                    </div>
                )
            }
        }

        if(this.state.authenticationError) {
            errorBox = (
                <div className={"errorBox"}>
                    <span>Authentication error</span>
                </div>
            )
        } else if(this.state.generalApiError) {
            errorBox = (
                <div className={"errorBox"}>
                    <span>Error, please try again later</span>
                </div>
            )
        }

        return (
            <div id={"textWidget"}>
                <div id={"textWidgetInput"}>
                    <div className={"text-input-prompt"}><span>Input your text:</span></div>
                    <div className={"text-input-area"}><textarea value={this.state.textAreaValue} onChange={this.handleTextAreaChange.bind(this)}/></div>
                    {authInputComponent}
                    <button onClick={this.handleButtonClick.bind(this)}>Submit</button>
                </div>
                {errorBox}
            </div>
        );
    }
}

export {TextSubmit}
