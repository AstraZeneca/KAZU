// @ts-ignore -- this library doesn't provide type annotations

import LabelStudio from "@heartexlabs/label-studio";
import React, {useEffect, useRef} from 'react';
import {Entity, KazuWebDocument, Section} from "../types/types";
import * as R from "rambda";


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


type LSPrediction = {

    "from_name": "tag",
    "id": string,
    "source": "$text",
    "to_name": "text",
    "ground_truth": true,
    "type": "rectanglelabels",
    "value": {
        "start": number,
        "end": number,
        "labels": string[]
    }
}


type LSTask = {
    annotations: [],
    predictions: {
        result: LSPrediction[],
        created_ago: string,
        model_version: string
    }[]
    id: number,
    data: {
        text: string
    }
}

type LSComponentProps = {
    kazuWebDocument?: KazuWebDocument;
}

type LSComponentState = {
    lsTask?: LSTask
}

class LSComponent extends React.Component<LSComponentProps, LSComponentState> {
    constructor(props: LSComponentProps) {
        super(props);
        this.state = {
            lsTask: this.props.kazuWebDocument ? LSComponent.lsTaskFromKazuDocument(this.props.kazuWebDocument) : undefined
        }
    }

    private static lsTaskFromKazuDocument(kazuDocument: KazuWebDocument): LSTask {
        return {
            annotations: [],
            data: {text: kazuDocument.sections[0].text},
            id: kazuDocument.sections[0].text.length,
            predictions: [{
                created_ago: "3 hours",
                result: LSComponent.lsPredictionsFromKazuDocument(kazuDocument),
                model_version: "model 1"
            }]
        }
    }

    private static lsPredictionsFromKazuDocument(kazuDocument: KazuWebDocument): LSPrediction[] {
        const ents: Entity[] = R.chain((section: Section) => section.entities)(kazuDocument.sections);
        return R.map((ent: Entity) => LSComponent.lsPredictionFromKazuEntity(ent))(ents);
    }

    private static lsPredictionFromKazuEntity(ent: Entity): LSPrediction {
        return {
            "from_name": "tag",
            "id": `${ent.match}_${ent.start}-${ent.end}`,
            "source": "$text",
            "to_name": "text",
            "ground_truth": true,
            "type": "rectanglelabels",
            "value": {
                "start": ent.start,
                "end": ent.end,
                "labels": [ent.entity_class]
            }
        }
    }

    private static labelConfig(text: string) {
        return `<View>
                    <Labels name="tag" toName="text">
                      <Label value="gene"></Label>
                      <Label value="anatomy"></Label>
                      <Label value="disease"></Label>
                      <Label value="drug"></Label>
                      <Label value="cell_line"></Label>
                    </Labels>
                    <Text name="text" value="$text"></Text>
                </View>`
    }

    render() {
        if (this.props.kazuWebDocument !== undefined) {
            const lsTask = LSComponent.lsTaskFromKazuDocument(this.props.kazuWebDocument);
            const conf = LSComponent.labelConfig(this.props.kazuWebDocument.sections[0].text)
            return <LabelStudioReact config={conf} task={lsTask} interfaces={[
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