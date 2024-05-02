import queryString from "query-string";
import { useLocation } from "react-router-dom";
import { datasetsSearchableContents } from "./Datasets.jsx";
import Highlighter from "react-highlight-words";
import Header from '../components/Header';
import Footer from '../components/Footer';

const convertToPageContents = ({ title, link, pageContents }) => {
    return pageContents.map((content, id) => {
        return {
            id,
            title,
            link,
            content,
        };
    });
};

const searchContent = (searchText) => {
    const searchTerms = searchText.split(" ");

    console.log("sldjkf")

    return [
        ...convertToPageContents({
            title: "datasets",
            link: "/datasets",
            pageContents: datasetsSearchableContents,
        }),
    ].filter(({ title, content }) => {
        return searchTerms.some((term) => {
            return (
                title.toLowerCase().includes(term.toLowerCase()) ||
                content.toLowerCase().includes(term.toLowerCase())
            );
        });
    });
};

const SearchResults = () => {
    const { search } = useLocation();
    const { q } = queryString.parse(search);
    console.log("page hit with", q)

    if (!q) {
        return null
    }


    const matchedContents = searchContent(q);
    console.log(
        "🚀 ~ file: SearchResults.jsx:68 ~ SearchResults ~ matchedContents:",
        matchedContents,
    );

    return (
        <div>
            <Header />
            <div className="container mx-auto mt-10 p-10 bg-emerald-200">
                <p className="text-xl">Users is trying to search: <span className="font-bold">{q}</span></p>
                <br></br>
                {matchedContents.map(({ title, link, content }, i) => {
                    return (
                        <div key={i}>
                            <a className="font-bold text-green-800" href={link}>{title}</a>
                            <p>
                                <Highlighter
                                    searchWords={q.split(" ")}
                                    autoEscape={true}
                                    textToHighlight={content}
                                />
                            </p>
                        </div>
                    );
                })}
            </div>
            <Footer />
        </div>
    );
};

export default SearchResults;