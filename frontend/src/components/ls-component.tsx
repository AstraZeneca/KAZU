// @ts-ignore -- this library doesn't provide type annotations
import LabelStudio from "@heartexlabs/label-studio";
import React, {useEffect, useRef} from 'react';
import {KazuLSResponse} from "../types/types";


type LabelStudioReactProps = {
    config: any;
    task: any;
    interfaces: any;
    user?: any;
}


const LabelStudioReact = (props: LabelStudioReactProps) => {
    const labelStudioContainerRef = useRef<HTMLDivElement>();
    const labelStudioRef = useRef();

    useEffect(() => {
        if (labelStudioContainerRef.current) {
            if (labelStudioRef.current) {
                // we have to manually destroy the LabelStudio instance -- it's a singleton
                // with static references to instances. The component doesn't update between re-renders without
                // this.

                // @ts-ignore
                labelStudioRef.current.destroy();
            }
            labelStudioRef.current = new LabelStudio(
                labelStudioContainerRef.current,
                props
            );
        }
    }, [props]);

    return (
        <div
            id="label-studio"
            ref={function (el) {
                labelStudioContainerRef.current = el == null? undefined : el
            }}
        />
    );
}

type LSComponentProps = {
    kazuLSAnnotations?: KazuLSResponse;
}

type LSComponentState = {

}

class LSComponent extends React.Component<LSComponentProps, LSComponentState> {
    constructor(props: LSComponentProps) {
        super(props);
        this.state = {

        }
    }

    render() {
        if (this.props.kazuLSAnnotations !== undefined) {
            const kazuLSAnnotations = this.props.kazuLSAnnotations
            const lsTasks = kazuLSAnnotations.ls_tasks
            const lsView = kazuLSAnnotations.ls_view
            return <LabelStudioReact config={lsView} task={lsTasks[0]} interfaces={[
                "panel",
                "update",
                "controls",
                "side-column",
                "annotations:menu",
                "annotations:add-new",
                "annotations:delete",
                "predictions:menu"
            ]}/>

        } else {
            return undefined
        }

    }
}

export {LSComponent}
