import { Component, type FormEvent, type ReactNode, type ChangeEvent } from 'react';
type DocsApiKeyInputState = {
    key: string;
    loadScript: boolean;
};
declare class DocsApiKeyInput extends Component<{}, DocsApiKeyInputState> {
    constructor(props: {});
    onInputChange: ({ target: { value }, }: ChangeEvent<HTMLInputElement>) => void;
    onFormSubmit: (event: FormEvent<HTMLFormElement>) => void;
    render(): ReactNode;
}
export default DocsApiKeyInput;
