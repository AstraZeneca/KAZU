import React from "react";
import {RawKazuDocument} from "../types/types";
import {JsonViewer} from "@textea/json-viewer";


type JsonViewState = {
    showJsonOutput: boolean;
}

type JsonViewProps = {
    rawDocument: RawKazuDocument
}

class JsonView extends React.Component<JsonViewProps, JsonViewState> {
    constructor(props: JsonViewProps) {
        super(props);
        this.state = {
            showJsonOutput: false
        }
    }


    handleToggleButtonClick() {
        this.setState((state) => {
            return {
                showJsonOutput: !state.showJsonOutput
            }
        })
    }

    renderToggleButton() {
        const buttonText = this.state.showJsonOutput? "Hide JSON": "Show JSON";
        return (
            <button id={"jsonToggleButton"} onClick={this.handleToggleButtonClick.bind(this)}>{buttonText}</button>
        )
    }

    render() {
        const toggleButton = this.renderToggleButton();
        const jsonViewComponent = this.state.showJsonOutput ? <JsonViewer value={this.props.rawDocument}/> : undefined;
        return(
            <div id={"jsonOutput"}>
                {toggleButton}
                {jsonViewComponent}
            </div>
        )
    }
}

export {JsonView}
